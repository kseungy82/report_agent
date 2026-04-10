from __future__ import annotations

"""
유틸 함수 모음.

현재는 `agents/report_agent.py`의 검증된 구현을 재사용(재export)합니다.
"""

from report_agent import (
    auto_slice_financials,
    extract_financial_periods,
    get_quarter,
    pick_reference_index,
    sort_period_keys,
    unit_multiplier_to_million,
)

__all__ = [
    "sort_period_keys",
    "pick_reference_index",
    "unit_multiplier_to_million",
    "get_quarter",
    "extract_financial_periods",
    "auto_slice_financials",
]
