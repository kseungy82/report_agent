from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from nodes import analyze_pdf


RouteName = Literal["report", "unsupported"]


@dataclass(frozen=True)
class RouteResult(TypedDict):
    route: RouteName
    report_text: str
    state: dict
    effective_pdf: str


@dataclass(frozen=True)
class RouterAgent:
    """
    상위 라우터 에이전트.
    현재 구현된 하위 에이전트는 `kanana_report_agent.ipynb` 기반 코드(report_agent) 1개입니다.
    """

    def choose(self, task: str) -> RouteName:
        t = (task or "").lower()
        if any(
            k in t
            for k in [
                "분기보고서",
                "사업보고서",
                "재무제표",
                "포괄손익",
                "지표",
                "영업이익",
                "매출",
                "pdf",
                "report",
            ]
        ):
            return "report"
        return "unsupported"

    def route_report(
        self,
        pdf_path: str,
        compare: Literal["QoQ", "YoY"] = "YoY",
        top_k: int = 5,
        use_reasoning: bool = True,
        slice_financial_statement: bool = True,
        work_dir: str | None = None,
    ) -> RouteResult:
        state, text, effective_pdf = analyze_pdf(
            pdf_path=pdf_path,
            compare=compare,
            top_k=top_k,
            use_reasoning=use_reasoning,
            slice_financial_statement=slice_financial_statement,
            work_dir=work_dir,
        )
        return {
            "route": "report",
            "report_text": text,
            "state": state,
            "effective_pdf": effective_pdf,
        }

    def route(
        self,
        task: str,
        pdf_path: str,
        compare: Literal["QoQ", "YoY"] = "YoY",
        top_k: int = 5,
        use_reasoning: bool = True,
        slice_financial_statement: bool = True,
        work_dir: str | None = None,
    ) -> RouteResult:
        route = self.choose(task)
        if route == "report":
            return self.route_report(
                pdf_path=pdf_path,
                compare=compare,
                top_k=top_k,
                use_reasoning=use_reasoning,
                slice_financial_statement=slice_financial_statement,
                work_dir=work_dir,
            )

        return {
            "route": "unsupported",
            "report_text": "현재 라우터는 보고서 분석 요청만 처리합니다.",
            "state": {},
            "effective_pdf": pdf_path,
        }
