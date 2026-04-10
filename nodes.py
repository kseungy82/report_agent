from __future__ import annotations

"""
LangGraph 노드/파이프라인 구성 모음.

핵심 노드 로직을 분리한 파일입니다.
"""

import math
import os
import re
from typing import Optional

from bs4 import BeautifulSoup
from langgraph.graph import END, StateGraph

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

# LLM/클라이언트 클래스는 report_agent 구현을 그대로 사용
from report_agent import GrowthReasoner, LLMClient, LLMTableExtractor, MetricSelector, UpstageDocumentParseClient

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
]


def node_upstage_parse(state: FGState, upstage: UpstageDocumentParseClient) -> FGState:
    warnings = list(state.get("warnings") or [])
    doc_bundle_by_pdf: dict[str, dict] = {}

    for path in state["pdf_paths"]:
        try:
            doc = upstage.parse(path)
            elements = doc.get("elements") or []
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
        except Exception as e:
            warnings.append(f"[{os.path.basename(path)}] Upstage 실패: {e}")

    return {**state, "doc_bundle_by_pdf": doc_bundle_by_pdf, "warnings": warnings}


def node_select_metrics(state: FGState, selector: MetricSelector) -> FGState:
    bundles = list(state["doc_bundle_by_pdf"].values())
    if not bundles:
        raise ValueError(f"문서 번들이 없습니다. warnings={state.get('warnings')}")
    selected_metrics = selector.run(bundles[0], CANDIDATE_METRICS)
    return {**state, "selected_metrics": selected_metrics}


def node_llm_extract(state: FGState, extractor: LLMTableExtractor) -> FGState:
    warnings = list(state.get("warnings") or [])
    llm_extraction_by_pdf: dict[str, LLMExtractionOutput] = {}
    for path, bundle in state["doc_bundle_by_pdf"].items():
        try:
            llm_extraction_by_pdf[path] = extractor.run(bundle, state["compare"], state["selected_metrics"])
        except Exception as e:
            warnings.append(f"[{os.path.basename(path)}] LLM 추출 실패: {e}")
    return {**state, "llm_extraction_by_pdf": llm_extraction_by_pdf, "warnings": warnings}


def node_merge_and_normalize(state: FGState) -> FGState:
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
    return {**state, "fin": fin, "warnings": warnings}


def node_compute_moves(state: FGState) -> FGState:
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

    return {**state, "now_period": now_p, "ref_period": ref_p, "ref_found": ref_idx is not None, "top_moves": top_moves[: state["top_k"]]}


def node_optional_reasoning(state: FGState, reasoner: Optional[GrowthReasoner]) -> FGState:
    if not reasoner:
        return state
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
        return {**state, "llm_reasoning": reasoner.run(inp)}
    except Exception as e:
        warnings = list(state.get("warnings") or [])
        warnings.append(f"Reasoning 실패: {e}")
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

