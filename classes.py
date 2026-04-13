from __future__ import annotations

"""
공통 스키마/타입 모음.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, Field

#Types / Schemas
Unit = Literal["원", "천원", "백만원", "천만원", "억원","UNKNOWN"]
CompareMode = Literal["QoQ", "YoY","UNKNOWN"] #분기 대비/전년동기 대비
PeriodKey = str  # "YYYYQn"

CANDIDATE_METRICS = [
    "영업이익(손실)", "법인세비용차감전순이익(손실)","법인세비용(수익)","당기순이익(손실)",
    "영업수익", "매출액", "순이자이익", "수수료수익",
    "매출원가", "매출총이익", "판매비와관리비",
    "영업이익",
    "법인세비용차감전순이익", "법인세비용", "당기순이익", "총포괄손익",
    "지배기업소유주지분순이익", "비지배지분순이익",
    "자산총계", "유동자산", "비유동자산", "유형자산", "무형자산",
    "부채총계", "유동부채", "비유동부채", "차입금",
    "자본총계", "지배기업소유주지분", "이익잉여금",
    "영업활동현금흐름", "투자활동현금흐름", "재무활동현금흐름",
    "수익(매출액)", "수익", "영업비용", "금융수익", "금융비용",
    "기타수익", "기타비용", "기타포괄손익"
]

#LLM이 기업 분석에 적합하다고 판단한 기업의 핵심 지표들
class MetricSelectionOutput(BaseModel):
    selected_metrics: List[str] = Field(description="제공된 보기 중 이 기업 분석에 가장 적합한 핵심 지표 5~10개")

#LLM이 추출한 보고서 기본 정보
class LLMExtractionOutput(BaseModel):
    unit_raw: str = Field(description="문서에 기록된 원본 단위")
    now_period: str = Field(description="당기 종료일 문자열")
    ref_period: str = Field(description="전기 종료일 문자열")
    now_period_key: str = Field(description="당기 PeriodKey (예: 2025Q1)")
    ref_period_key: str = Field(description="전기 PeriodKey (예: 2024Q1)")

    items: Dict[str, Dict[str, Union[str, int, float, None]]] = Field(
        description="PeriodKey별 지표 데이터"
    )
#단위 '백만원'으로 통일 후 시계열로 정렬한 데이터
class NormalizedFinancials(BaseModel):
    unit: Unit
    periods: List[PeriodKey] 
    pl: Dict[str, List[float]]  

#지표 변화 요약 (전년 또는 전분기 대비 증감과 증감률 계산 결과 저장)
class TopMove(BaseModel):
    metric: str
    now_period: PeriodKey
    ref_period: PeriodKey
    now: float   #이번
    ref: float   #비교
    delta: float #변화량
    delta_pct: Optional[float] = None  #변화율
    flip_flag: bool = False  #흑자,적자 전환 여부

#GrowthReasoningOutput에 보낼 내용 정리
class GrowthReasoningInput(BaseModel):
    unit: Unit   #화폐 단위
    periods: List[PeriodKey]
    metrics_timeseries: Dict[str, List[float]]  
    top_moves: List[TopMove]
    compare: CompareMode   #분기별/전년 동기 (QoQ/YoY)
    now_period: PeriodKey
    ref_period: Optional[PeriodKey]

class GrowthReasoningOutput(BaseModel):
    growth_trajectory: str
    key_changes: List[str]
    caveats: List[str]  
    summary_table: Optional[str]


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

__all__ = [
    "Unit",
    "CompareMode",
    "PeriodKey",
    "CANDIDATE_METRICS",
    "MetricSelectionOutput",
    "LLMExtractionOutput",
    "NormalizedFinancials",
    "TopMove",
    "GrowthReasoningInput",
    "GrowthReasoningOutput",
    "FGState",
]

