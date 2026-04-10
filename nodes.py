from __future__ import annotations
import math
import os
import re
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
import torch
from bs4 import BeautifulSoup
from langgraph.graph import END, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer

from classes import (
    CANDIDATE_METRICS,
    FGState,
    GrowthReasoningInput,
    GrowthReasoningOutput,
    LLMExtractionOutput,
    NormalizedFinancials,
    TopMove,
)
from utils import pick_reference_index, sort_period_keys, unit_multiplier_to_million

from config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "node_upstage_parse",
    "node_select_metrics",
    "node_llm_extract",
    "node_merge_and_normalize",
    "node_compute_moves",
    "node_optional_reasoning",
    "build_graph",
    "run_pipeline",
    "render_report",
    "MetricSelector",
    "LLMTableExtractor",
    "GrowthReasoner",
    "analyze_pdf",
]


class UpstageDocumentParseClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.upstage_api_key
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY is required. Put it in environment or .env.")
        self.base_url = "https://api.upstage.ai/v1/document-digitization"

    def parse(self, pdf_path: str, timeout: int = 600) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": "document-parse-260128",
            "ocr": "auto",
            "chart_recognition": "true",
            "coordinates": "true",
            "output_formats": '["html"]',
            "base64_encoding": '["figure"]',
        }
        logger.info("[Upstage] parse start: file=%s timeout=%ss", os.path.basename(pdf_path), timeout)
        started_at = datetime.now()
        with open(pdf_path, "rb") as f:
            files = {"document": f}
            try:
                resp = requests.post(self.base_url, headers=headers, files=files, data=data, timeout=timeout)
                elapsed = (datetime.now() - started_at).total_seconds()
                logger.info("[Upstage] parse response: status=%s elapsed=%.1fs", resp.status_code, elapsed)
                resp.raise_for_status()
                payload = resp.json()
                logger.info("[Upstage] parse success: keys=%s", list(payload.keys()))
                return payload
            except requests.Timeout:
                elapsed = (datetime.now() - started_at).total_seconds()
                logger.exception("[Upstage] parse timeout after %.1fs", elapsed)
                raise
            except requests.RequestException:
                elapsed = (datetime.now() - started_at).total_seconds()
                logger.exception("[Upstage] parse request failed after %.1fs", elapsed)
                raise


class LLMClient:
    def __init__(self, model_id: str, use_cpu: bool = False):
        self.device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("%s 로딩 중... device=%s", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if "<|user|>" in content:
            content = content.split("<|user|>")[0].strip()
        if "<|assistant|>" in content:
            content = content.split("<|assistant|>")[0].strip()
        content = content.replace("```json", "").replace("```", "").strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        return content[start:end] if start != -1 and end > start else content


class MetricSelector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, doc_bundle: dict, candidates: list[str]) -> list[str]:
        from classes import MetricSelectionOutput

        schema = MetricSelectionOutput.model_json_schema()
        system = (
            "너는 10년차 재무 분석 전문가야.\n"
            f"제공된 문서를 분석하여, 기업의 재무 건전성과 성장성을 가장 잘 보여주는 핵심 지표를 최소 5개 선정해.\n"
            "단, 아래의 조건을 엄격히 준수할 것:\n"
            "1. '포괄손익계산서'에 적혀있는 지표명 그대로 추출해. 괄호 있는 경우 괄호 안 글자까지 그대로 추출해.\n"
            "2. '포괄손익계산서'에 적힌 텍스트와 100% 일치하는 지표이면서, 동시에 다음 [후보 지표 목록]에 포함된 지표만 선택할 것.\n"
            f"  [후보 지표 목록]: {candidates}\n"
            "3. 출력은 반드시 제공된 스키마와 일치하는 JSON 형태여야 하며, JSON 내부에 '//' 형태의 주석이나 부가 설명은 절대 포함하지 말 것.\n"
        )
        user = f"DOCUMENT_BUNDLE:\n{json.dumps(doc_bundle, ensure_ascii=False, indent=2)}\n\nSCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
        raw = self.llm.generate(system=system, user=user)
        return MetricSelectionOutput(**json.loads(raw)).selected_metrics


class LLMTableExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, doc_bundle: dict, compare: str, selected_metrics: list[str]) -> LLMExtractionOutput:
        from classes import LLMExtractionOutput
        from utils import extract_financial_periods

        schema = LLMExtractionOutput.model_json_schema()
        now_date, now_key, now_term, ref_date, ref_key, ref_term = extract_financial_periods(doc_bundle)
        system = f"""
너는 제공된 문서에서 재무제표 핵심 수치만 선별해 정확한 수치를 추출하는 수석 재무 분석가다.

[분석 대상 기간]
아래 4개의 Key는 반드시 아래 지정해준 값으로 JSON에 넣어.
- 'now_period_key': '{now_key}'
- 'now_period': '{now_date}' (표에서 데이터를 찾을 때 참고할 열 이름표: {now_term}')
- 'ref_period_key': '{ref_key}'
- 'ref_period": '{ref_date}' (표에서 데이터를 찾을 때 참고할 열 이름표: '{ref_term}')

[목표]
- 탐색 범위: 제공된 데이터의 `texts`(일반 텍스트 영역)와 `tables`(표 영역) 두 곳을 모두 탐색해라.
- 추출 대상: 위 두 영역을 종합하여 '문서에 기록된 원본 단위(unit_raw)', '당기기간 및 데이터', '비교기간 및 데이터'를 찾아 추출해라.
- 결과를 반드시 제공된 스키마와 일치하는 JSON으로만 출력해라.

[데이터 추출 절대 원칙: 원본 100% 일치]
1. 지표명(Key): 반드시 문서에 적힌 글자 그대로 출력해.
2. 수치(Value): 단 한 글자도 바꾸지 말고 표에 보이는 그대로 복사해서 출력해.

[지표명 복사 예시 (형식 변형 절대 금지)]
- 괄호() 없는 원본: 영업이익 (O) / 영업이익(손실) (X)
- 괄호() 있는 원본: 금융수익(손실) (O) / 금융수익 (X)

[수치 복사 예시 (형식 변형 절대 금지)]
- 쉼표(,) 유지: 489,341,873 (O) / 489341873 (X)
- 괄호() 없는 원본: 1234 (O) / (1234) (X)
- 괄호() 있는 원본: (1470) (O) / 1470 (X)

[값 추출 규칙]
1. 반드시 {selected_metrics}에 포함된 지표들만 추출해.
3. 당기({now_term}) 데이터는 지표명 우측에 있는 **첫 번째 숫자**('3개월' 열)를 추출해.
4. 전기({ref_term}) 데이터는 지표명 우측에 있는 **세 번째 숫자**('3개월' 열)를 추출해.
5. 'items' 딕셔너리의 최상위 Key는 반드시 위에서 만든 PeriodKey여야 해.
6. 모든 수치 값은 반드시 문자열로 출력해. 숫자형으로 출력하지 마.
7. 모든 수치 반드시 맨 끝자리까지 정확하게 출력해.

[자체 검증 - 반드시 수행]
- 최종 출력은 반드시 스키마와 일치하는 JSON만 출력하라.
"""
        user = (
            f"DOCUMENT_BUNDLE:\n{json.dumps(doc_bundle, ensure_ascii=False, indent=2)}\n\n"
            f"SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n"
            f"COMPARE_HINT: {compare}"
        )
        raw = self.llm.generate(system=system, user=user)
        return LLMExtractionOutput(**json.loads(raw))


class GrowthReasoner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, inp: GrowthReasoningInput) -> GrowthReasoningOutput:
        now_p = inp.now_period
        ref_p = inp.ref_period or "UNKNOWN"
        unit = inp.unit
        table_rows = [
            f"| 지표 | {ref_p} ({unit}) | {now_p} ({unit}) | 변동액 ({unit}) | 증감률 (%) |",
            "|---|---|---|---|---|",
        ]
        analysis_hints: list[str] = []
        for m in inp.top_moves:
            pct_str = f"{m.delta_pct * 100:.1f}%" if m.delta_pct is not None else "n/a"
            table_rows.append(f"| {m.metric} | {m.ref:,.2f} | {m.now:,.2f} | {m.delta:,.2f} | {pct_str} |")
            status = "흑자전환" if m.flip_flag and m.now > 0 else ("적자/악화" if m.delta < 0 else "성장/개선")
            analysis_hints.append(f"- {m.metric}: {m.ref:,.0f} -> {m.now:,.0f} (변동: {m.delta:,.0f}, {pct_str}) [{status}]")
        perfect_markdown_table = "\n".join(table_rows)
        system = (
            "너는 기업 실적을 분석하는 10년차 재무 분석 전문가야. 모든 답변은 한국어로 작성해.\n"
            "제공된 문서를 바탕으로 아래 양식에 맞춰서 텍스트로만 답변해. 절대 JSON 형식({ })이나 표를 쓰지 마.\n\n"
            "[제약 조건]\n"
            "1. 반드시 '포괄손익계산서'에 있는 5개의 지표에 대해서만 작성해. 문서에 없는 지표명 사용 금지.\n"
            "2. 중복된 내용 반복 출력 금지.\n\n"
            "[총평]\n"
            "(기업의 실적 변동에 대한 총평을 금융 전문가 말투로 작성)\n\n"
            "[지표분석]\n"
            "지표명 출력: 괄호와 괄호 안의 글자는 무조건 삭제하고 출력해. (예: '영업이익(손실)' -> '영업이익')\n"
            "분석 내용 작성 시 괄호 안 글자는 고려하지 말고 분석. (예: '영업이익(손실)' -> '영업이익'으로 생각하고 분석)\n"
            "분석 시 퍼센트 언급 가능. 금액 언급 금지.\n"
            "제공된 지표 모두에 대해 아래와 같은 리스트 형식으로 금융 전문가 말투의 분석 요약 작성\n"
            "- [해당 지표명]: (분석 내용 서술형으로 작성)\n"
            "\n[특이사항]\n"
            "- (모든 지표 고려하여 우려되는 점이나 모니터링이 필요한 점을 금융전문가 말투로 작성. 단, 추측 금지, 오직 사실에 근거해 작성.)"
        )
        raw = self.llm.generate(system=system, user="실적 데이터:\n" + "\n".join(analysis_hints)).strip()
        growth_traj = raw
        key_changes: list[str] = []
        caveats: list[str] = []
        if "[총평]" in raw:
            growth_traj = raw.split("[총평]", 1)[1].split("[지표분석]", 1)[0].strip()
        if "[지표분석]" in raw:
            part = raw.split("[지표분석]", 1)[1].split("[특이사항]", 1)[0].strip()
            key_changes = [ln.strip().lstrip("-").strip() for ln in part.splitlines() if ln.strip()]
        if "[특이사항]" in raw:
            part = raw.split("[특이사항]", 1)[1].strip()
            caveats = [ln.strip().lstrip("-").strip() for ln in part.splitlines() if ln.strip()]
        return GrowthReasoningOutput(
            growth_trajectory=growth_traj or "분석 요약 생성 실패",
            key_changes=key_changes or [h for h in analysis_hints],
            caveats=caveats,
            summary_table=perfect_markdown_table,
        )


def node_upstage_parse(state: FGState, upstage: UpstageDocumentParseClient) -> FGState:
    warnings = list(state.get("warnings") or [])
    doc_bundle_by_pdf: dict[str, dict] = {}

    for path in state["pdf_paths"]:
        try:
            logger.info("[Pipeline] upstage_parse begin: %s", path)
            doc = upstage.parse(path)
            elements = doc.get("elements") or []
            logger.info("[Pipeline] upstage_parse elements: count=%s", len(elements))
            all_htmls = []
            all_texts = []

            for el in elements:
                content = el.get("content", {})
                if "html" in content:
                    raw_html = content["html"]
                    soup = BeautifulSoup(raw_html, "html.parser")
                    for tr in soup.find_all("tr"):
                        cells = tr.find_all(["td", "th"])
                        if len(cells) >= 5:
                            cells[4].extract()
                            cells[2].extract()
                        elif len(cells) == 4:
                            text_check = cells[1].get_text(strip=True)
                            if "누적" in text_check:
                                cells[3].extract()
                                cells[1].extract()
                    safe_html = str(soup)
                    all_htmls.append({"html": safe_html.strip()[:12000]})
                elif "text" in content:
                    all_texts.append(content["text"].strip())

            doc_bundle_by_pdf[path] = {
                "file": os.path.basename(path),
                "texts": all_texts,
                "tables": all_htmls,
            }
            logger.info(
                "[Pipeline] upstage_parse done: file=%s texts=%s tables=%s",
                os.path.basename(path),
                len(all_texts),
                len(all_htmls),
            )
        except Exception as e:
            warnings.append(f"[{os.path.basename(path)}] Upstage 실패: {e}")
            logger.exception("[Pipeline] upstage_parse failed: %s", os.path.basename(path))

    return {**state, "doc_bundle_by_pdf": doc_bundle_by_pdf, "warnings": warnings}


def node_select_metrics(state: FGState, selector: MetricSelector) -> FGState:
    started_at = datetime.now()
    logger.info("[Pipeline] select_metrics begin")
    bundles = list(state["doc_bundle_by_pdf"].values())
    if not bundles:
        raise ValueError(f"문서 번들이 없습니다. warnings={state.get('warnings')}")
    selected_metrics = selector.run(bundles[0], CANDIDATE_METRICS)
    elapsed = (datetime.now() - started_at).total_seconds()
    logger.info("[Pipeline] select_metrics done: count=%s elapsed=%.1fs", len(selected_metrics), elapsed)
    return {**state, "selected_metrics": selected_metrics}


def node_llm_extract(state: FGState, extractor: LLMTableExtractor) -> FGState:
    started_at = datetime.now()
    logger.info("[Pipeline] llm_extract begin: pdf_count=%s", len(state["doc_bundle_by_pdf"]))
    warnings = list(state.get("warnings") or [])
    llm_extraction_by_pdf: dict[str, LLMExtractionOutput] = {}
    for path, bundle in state["doc_bundle_by_pdf"].items():
        try:
            one_started = datetime.now()
            logger.info("[Pipeline] llm_extract per-pdf begin: %s", os.path.basename(path))
            llm_extraction_by_pdf[path] = extractor.run(bundle, state["compare"], state["selected_metrics"])
            one_elapsed = (datetime.now() - one_started).total_seconds()
            logger.info("[Pipeline] llm_extract per-pdf done: %s elapsed=%.1fs", os.path.basename(path), one_elapsed)
        except Exception as e:
            warnings.append(f"[{os.path.basename(path)}] LLM 추출 실패: {e}")
            logger.exception("[Pipeline] llm_extract per-pdf failed: %s", os.path.basename(path))
    elapsed = (datetime.now() - started_at).total_seconds()
    logger.info("[Pipeline] llm_extract done: success=%s elapsed=%.1fs", len(llm_extraction_by_pdf), elapsed)
    return {**state, "llm_extraction_by_pdf": llm_extraction_by_pdf, "warnings": warnings}


def node_merge_and_normalize(state: FGState) -> FGState:
    started_at = datetime.now()
    logger.info("[Pipeline] merge_normalize begin")
    warnings = list(state.get("warnings") or [])
    merged_data: dict[str, dict[str, float]] = {}
    unit_raw = "UNKNOWN"

    for ext in state["llm_extraction_by_pdf"].values():
        if unit_raw == "UNKNOWN" and ext.unit_raw != "UNKNOWN":
            unit_raw = ext.unit_raw

    unit_raw = str(unit_raw).strip().replace("(", "").replace(")", "").replace("단위", "").replace(":", "").replace(" ", "")
    scale = unit_multiplier_to_million(unit_raw)
    allowed_metrics = {m.strip() for m in state["selected_metrics"]}

    for path, ext in state["llm_extraction_by_pdf"].items():
        for pk, metrics_dict in ext.items.items():
            if not re.match(r"^\d{4}Q[1-4]$", pk):
                continue
            merged_data.setdefault(pk, {})

            for metric, value in metrics_dict.items():
                if value is None:
                    continue
                try:
                    clean_metric = str(metric).strip()
                    if clean_metric not in allowed_metrics:
                        continue
                    clean_val = str(value).strip().replace(",", "").replace(" ", "")
                    if not clean_val:
                        continue

                    if clean_val.startswith("(") and clean_val.endswith(")"):
                        clean_val = "-" + clean_val[1:-1]
                    elif clean_val.startswith("△"):
                        clean_val = "-" + clean_val[1:]
                    elif clean_val.startswith("-(") and clean_val.endswith(")"):
                        clean_val = "-" + clean_val[2:-1]

                    merged_data[pk][clean_metric] = float(clean_val) * scale
                except Exception as e:
                    warnings.append(
                        f"[{os.path.basename(path)}] 값 변환 실패: pk={pk}, metric={metric}, value={value}, err={repr(e)}"
                    )

    periods = sort_period_keys(list(merged_data.keys()))
    pl_series = {}
    for m in state["selected_metrics"]:
        m_strip = m.strip()
        pl_series[m_strip] = [merged_data.get(pk, {}).get(m_strip, math.nan) for pk in periods]

    fin = NormalizedFinancials(unit="백만원", periods=periods, pl=pl_series)
    elapsed = (datetime.now() - started_at).total_seconds()
    logger.info("[Pipeline] merge_normalize done: periods=%s metrics=%s elapsed=%.1fs", len(periods), len(pl_series), elapsed)
    return {**state, "fin": fin, "warnings": warnings}


def node_compute_moves(state: FGState) -> FGState:
    started_at = datetime.now()
    logger.info("[Pipeline] compute_moves begin")
    fin = state["fin"]
    warnings = list(state.get("warnings") or [])

    if not fin.periods:
        warnings.append("추출된 유효한 기간 데이터가 없습니다.")
        return {
            **state,
            "now_period": "UNKNOWN",
            "ref_period": None,
            "ref_found": False,
            "top_moves": [],
            "warnings": warnings,
        }

    now_idx = len(fin.periods) - 1
    now_p = fin.periods[now_idx]
    ref_idx = pick_reference_index(fin.periods, now_idx, state["compare"])
    ref_p = fin.periods[ref_idx] if ref_idx is not None else None

    top_moves: list[TopMove] = []
    if ref_idx is not None and ref_p is not None:
        for metric, values in fin.pl.items():
            now, ref = values[now_idx], values[ref_idx]
            if math.isnan(now) or math.isnan(ref):
                continue
            d = now - ref
            pct = None if ref == 0 else d / ref
            top_moves.append(
                TopMove(
                    metric=metric,
                    now_period=now_p,
                    ref_period=ref_p,
                    now=now,
                    ref=ref,
                    delta=d,
                    delta_pct=pct,
                    flip_flag=(now >= 0 > ref) or (ref >= 0 > now),
                )
            )
        top_moves.sort(key=lambda x: (abs(x.delta_pct or 0.0), abs(x.delta)), reverse=True)

    result = {**state, "now_period": now_p, "ref_period": ref_p, "ref_found": ref_idx is not None, "top_moves": top_moves[: state["top_k"]]}
    elapsed = (datetime.now() - started_at).total_seconds()
    logger.info("[Pipeline] compute_moves done: top_moves=%s elapsed=%.1fs", len(result["top_moves"]), elapsed)
    return result


def node_optional_reasoning(state: FGState, reasoner: Optional[GrowthReasoner]) -> FGState:
    if not reasoner:
        logger.info("[Pipeline] optional_reasoning skipped")
        return state
    started_at = datetime.now()
    logger.info("[Pipeline] optional_reasoning begin")
    inp = GrowthReasoningInput(
        unit=state["fin"].unit,
        periods=state["fin"].periods,
        metrics_timeseries=state["fin"].pl,
        top_moves=state["top_moves"],
        compare=state["compare"],
        now_period=state["now_period"],
        ref_period=state.get("ref_period"),
    )
    try:
        out = {**state, "llm_reasoning": reasoner.run(inp)}
        elapsed = (datetime.now() - started_at).total_seconds()
        logger.info("[Pipeline] optional_reasoning done elapsed=%.1fs", elapsed)
        return out
    except Exception as e:
        warnings = list(state.get("warnings") or [])
        warnings.append(f"Reasoning 실패: {e}")
        logger.exception("[Pipeline] optional_reasoning failed")
        return {**state, "llm_reasoning": None, "warnings": warnings}


def render_report(state: dict) -> str:
    fin = state["fin"]
    lines = [
        f"[단위]: {fin.unit}, [비교]: {state['compare']}",
        f"[선택된 지표] {', '.join(state['selected_metrics'])}",
    ]
    lines.append("\n1) 핵심 지표 변동 (Top Moves)")
    if state.get("top_moves"):
        for m in state["top_moves"]:
            pct = f"{m.delta_pct * 100:.1f}%" if m.delta_pct is not None else "n/a"
            lines.append(f"- {m.metric}: Δ {m.delta:,.0f} ({pct})")
    else:
        lines.append("- 추출된 지표 변동 내역이 없습니다.")

    if state.get("llm_reasoning"):
        rr: GrowthReasoningOutput = state["llm_reasoning"]
        lines.append("\n2) 분석 요약")
        lines.append(f"- 총평: {rr.growth_trajectory}")
        if rr.summary_table:
            lines.append("\n[핵심 지표 변화 요약 표]")
            lines.append(rr.summary_table)
            lines.append("")
        for b in rr.key_changes or []:
            lines.append(f"  • {b}")
        if rr.caveats:
            lines.append("")
            lines.append("- 특이사항/주의:")
            for c in rr.caveats:
                lines.append(f"  • {c}")

    if state.get("warnings"):
        lines.append("\n[내부 경고/에러 로그]")
        for w in state["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)


def build_graph(
    selector: MetricSelector,
    extractor: LLMTableExtractor,
    upstage: UpstageDocumentParseClient,
    reasoner: Optional[GrowthReasoner] = None,
):
    g = StateGraph(FGState)
    g.add_node("upstage_parse", lambda s: node_upstage_parse(s, upstage))
    g.add_node("select_metrics", lambda s: node_select_metrics(s, selector))
    g.add_node("llm_extract", lambda s: node_llm_extract(s, extractor))
    g.add_node("merge_normalize", node_merge_and_normalize)
    g.add_node("compute_moves", node_compute_moves)
    g.add_node("optional_reasoning", lambda s: node_optional_reasoning(s, reasoner))

    g.set_entry_point("upstage_parse")
    g.add_edge("upstage_parse", "select_metrics")
    g.add_edge("select_metrics", "llm_extract")
    g.add_edge("llm_extract", "merge_normalize")
    g.add_edge("merge_normalize", "compute_moves")
    g.add_edge("compute_moves", "optional_reasoning")
    g.add_edge("optional_reasoning", END)
    return g.compile()


def run_pipeline(
    pdf_paths: list[str],
    llm: LLMClient,
    upstage: UpstageDocumentParseClient,
    compare: str = "YoY",
    top_k: int = 5,
    use_reasoning: bool = False,
):
    app = build_graph(
        MetricSelector(llm),
        LLMTableExtractor(llm),
        upstage,
        GrowthReasoner(llm) if use_reasoning else None,
    )
    final_state = app.invoke({"pdf_paths": pdf_paths, "compare": compare, "top_k": top_k, "warnings": []})
    return final_state, render_report(final_state)


@dataclass
class ModelBundle:
    llm: LLMClient
    upstage: UpstageDocumentParseClient


_bundle_lock = threading.Lock()
_bundle: ModelBundle | None = None


def get_model_bundle() -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle
    with _bundle_lock:
        if _bundle is not None:
            return _bundle
        if settings.huggingfacehub_api_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.huggingfacehub_api_token
        upstage = UpstageDocumentParseClient()
        llm = LLMClient(model_id=settings.kanana_model_id)
        _bundle = ModelBundle(llm=llm, upstage=upstage)
        return _bundle


def analyze_pdf(
    pdf_path: str,
    compare: str = "YoY",
    top_k: int = 5,
    use_reasoning: bool = True,
    slice_financial_statement: bool = True,
    work_dir: str | None = None,
):
    from utils import auto_slice_financials

    bundle = get_model_bundle()
    effective_pdf = pdf_path
    if slice_financial_statement:
        base_dir = work_dir or os.path.dirname(os.path.abspath(pdf_path))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sliced = os.path.join(base_dir, f"temp_sliced_finance_{ts}.pdf")
        effective_pdf = auto_slice_financials(pdf_path, sliced)

    state, text = run_pipeline(
        pdf_paths=[effective_pdf],
        llm=bundle.llm,
        upstage=bundle.upstage,
        compare=compare,
        top_k=top_k,
        use_reasoning=use_reasoning,
    )
    return state, text, effective_pdf

