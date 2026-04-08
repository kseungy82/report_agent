import os
import re
import math
from typing import Any, Dict, List, Optional, TypedDict
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END
from config import CompareMode
from utils import PeriodKey, NormalizedFinancials, TopMove, unit_multiplier_to_million, sort_period_keys, pick_reference_index, LLMExtractionOutput, GrowthReasoningOutput
from classes import UpstageDocumentParseClient, MetricSelector, LLMTableExtractor, GrowthReasoner
import logging

class FGState(TypedDict, total=False):
    pdf_paths: List[str]
    compare: CompareMode
    top_k: int

    doc_bundle_by_pdf: Dict[str, Dict[str, Any]]
    selected_metrics: List[str]  # Pass 1 결과물
    llm_extraction_by_pdf: Dict[str, LLMExtractionOutput]

    fin: NormalizedFinancials
    now_period: PeriodKey
    ref_period: Optional[PeriodKey]
    ref_found: bool
    top_moves: List[TopMove]
    llm_reasoning: Optional[GrowthReasoningOutput]
    warnings: List[str]
  
def node_upstage_parse(state: FGState, upstage: UpstageDocumentParseClient) -> FGState:
    warnings = list(state.get("warnings") or [])
    doc_bundle_by_pdf = {}

    for path in state["pdf_paths"]:
        try:
            doc = upstage.parse(path)
            elements = doc.get("elements") or []

            all_htmls = []
            all_texts = []  # 표 외의 모든 텍스트 정보를 담을 바구니

            for i, el in enumerate(elements):
                content = el.get("content", {})
                category = el.get("category", "") # 요소의 종류 (text, table, heading 등)

                # 표(table)인 경우: HTML 정제 후 저장
                if "html" in content:
                    raw_html = content["html"]
                    soup = BeautifulSoup(raw_html, "html.parser")

                    # '누적' 열 삭제 로직 
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
                    # 모든 표를 저장하되, 너무 길면 자름
                    all_htmls.append({"html": safe_html.strip()[:12000]})

                # 표가 아닌 모든 텍스트 요소(text, heading, caption 등) 수집
                elif "text" in content:
                    all_texts.append(content["text"].strip())

            # 수집된 모든 데이터를 번들에 담기
            doc_bundle_by_pdf[path] = {
                "file": os.path.basename(path),
                "texts": all_texts,  
                "tables": all_htmls
            }

        except Exception as e:
            warnings.append(f"[{os.path.basename(path)}] Upstage 실패: {e}")

    return {**state, "doc_bundle_by_pdf": doc_bundle_by_pdf, "warnings": warnings}

def node_select_metrics(state: FGState, selector: MetricSelector) -> FGState:
    """Pass 1: 첫 번째 PDF(최신)를 기준으로 핵심 지표를 선택합니다."""
    bundles = list(state["doc_bundle_by_pdf"].values())
    if not bundles:
      logging.info("\n[Upstage 파싱 실패 상세 원인]")
      logging.info(state.get("warnings"))
      raise ValueError("문서 번들이 없습니다.")

    selected_metrics = selector.run(bundles[0], CANDIDATE_METRICS)
    return {**state, "selected_metrics": selected_metrics}


def node_llm_extract(state: FGState, extractor: LLMTableExtractor) -> FGState:
    """Pass 2: 선택된 지표만 모든 PDF에서 추출합니다."""
    warnings = list(state.get("warnings") or [])
    llm_extraction_by_pdf = {}
    for path, bundle in state["doc_bundle_by_pdf"].items():
        try:
            llm_extraction_by_pdf[path] = extractor.run(bundle, state["compare"], state["selected_metrics"])
        except Exception as e:
            warnings.append(f"[{os.path.basename(path)}] LLM 추출 실패: {e}")
    return {**state, "llm_extraction_by_pdf": llm_extraction_by_pdf, "warnings": warnings}

def node_merge_and_normalize(state: FGState) -> FGState:
    warnings = list(state.get("warnings") or [])
    merged_data: Dict[PeriodKey, Dict[str, float]] = {}
    unit_raw = "UNKNOWN"

    # 메타 정보 수집
    for _, ext in state["llm_extraction_by_pdf"].items():
        if unit_raw == "UNKNOWN" and ext.unit_raw != "UNKNOWN":
            unit_raw = ext.unit_raw

    unit_raw = str(unit_raw).strip().replace("(", "").replace(")", "").replace("단위", "").replace(":", "").replace(" ", "")
    scale = unit_multiplier_to_million(unit_raw)

    allowed_metrics = {m.strip() for m in state["selected_metrics"]}

    # 기간별 지표 병합
    for path, ext in state["llm_extraction_by_pdf"].items():
        for pk, metrics_dict in ext.items.items():
            if not re.match(r"^\d{4}Q[1-4]$", pk):
                continue

            if pk not in merged_data:
                merged_data[pk] = {}

            for metric, value in metrics_dict.items():
                if value is None:
                    continue

                try:
                    clean_metric = metric.strip()

                    # 선택된 지표만 유지 
                    if clean_metric not in allowed_metrics:
                        continue

                    clean_val = str(value).strip()
                    clean_val = clean_val.replace(",", "").replace(" ","")

                    if not clean_val:
                        continue

                    # (100) -> -100
                    if clean_val.startswith("(") and clean_val.endswith(")"):
                        clean_val = "-" + clean_val[1:-1]
                    # △100 -> -100
                    elif clean_val.startswith("△"):
                        clean_val = "-" + clean_val[1:]
                    # 혹시 LLM이 -(100) 같이 줬을 경우
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
        pl_series[m_strip] = [
            merged_data.get(pk, {}).get(m_strip, math.nan)
            for pk in periods
        ]

    fin = NormalizedFinancials(
        unit="백만원",
        periods=periods,
        pl=pl_series,
    )

    return {**state, "fin": fin, "warnings": warnings}

def node_compute_moves(state: FGState) -> FGState:
    fin = state["fin"]
    warnings = list(state.get("warnings") or [])

    # 유효한 기간 데이터가 아예 없는 경우
    if not fin.periods:
        warnings.append("추출된 유효한 기간 데이터가 없습니다. (LLM의 형식 위반 또는 누적 데이터 오인 필터링)")
        return {
            **state,
            "now_period": "UNKNOWN",
            "ref_period": None,
            "ref_found": False,
            "top_moves": [],
            "warnings": warnings
        }
    now_idx = len(fin.periods) - 1
    now_p = fin.periods[now_idx]
    ref_idx = pick_reference_index(fin.periods, now_idx, state["compare"])
    ref_p = fin.periods[ref_idx] if ref_idx is not None else None

    top_moves = []
    if ref_idx is not None:
        for metric, values in fin.pl.items():
            now, ref = values[now_idx], values[ref_idx]
            if math.isnan(now) or math.isnan(ref): continue
            d = now - ref
            pct = None if ref == 0 else d / ref
            top_moves.append(TopMove(metric=metric, now_period=now_p, ref_period=ref_p, now=now, ref=ref, delta=d, delta_pct=pct, flip_flag=(now >= 0 > ref) or (ref >= 0 > now)))
        top_moves.sort(key=lambda x: (abs(x.delta_pct or 0.0), abs(x.delta)), reverse=True)

    return {**state, "now_period": now_p, "ref_period": ref_p, "ref_found": ref_idx is not None, "top_moves": top_moves[:state["top_k"]]}


def node_optional_reasoning(state: FGState, reasoner: Optional[GrowthReasoner]) -> FGState:
    if not reasoner: return state
    inp = GrowthReasoningInput(
        unit=state["fin"].unit,
        periods=state["fin"].periods, metrics_timeseries=state["fin"].pl,
        top_moves=state["top_moves"], compare=state["compare"], now_period=state["now_period"], ref_period=state.get("ref_period")
    )
    try:
      return {**state, "llm_reasoning": reasoner.run(inp)}
    except Exception as e:
      warnings = list(state.get("warnings") or [])
      warnings.append(f"Reasoning 실패: {e}")
      return {**state, "llm_reasoning": None}

def render_report(state: dict) -> str:
    fin = state["fin"]
    lines = [
        f"[단위]: {fin.unit}, [비교]: {state['compare']}",
        f"[선택된 지표] {', '.join(state['selected_metrics'])}"
    ]
    # 핵심 지표 변동 (Top Moves)
    lines.append("\n1) 핵심 지표 변동 (Top Moves)")
    if state.get("top_moves"):
        for m in state["top_moves"]:
            pct = f"{m.delta_pct*100:.1f}%" if m.delta_pct is not None else "n/a"
            lines.append(f"- {m.metric}: Δ {m.delta:,.0f} ({pct})")
    else:
        lines.append("- 추출된 지표 변동 내역이 없습니다.")

    # 분석 요약
    if state.get("llm_reasoning"):
        rr = state["llm_reasoning"]
        lines.append("\n2) 분석 요약")
        lines.append(f"- 총평: {rr.growth_trajectory}")

        # 파이썬이 완벽하게 생성한 표 삽입
        if rr.summary_table:
            lines.append("\n[핵심 지표 변화 요약 표]")
            lines.append(rr.summary_table)
            lines.append("")

        # LLM이 작성한 분석글 
        if rr.key_changes:
            for b in rr.key_changes:
                lines.append(f"  • {b}")

        # 주의사항
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

def build_graph(selector: MetricSelector, extractor: LLMTableExtractor, upstage: UpstageDocumentParseClient, reasoner: Optional[GrowthReasoner] = None):
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
