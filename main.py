from __future__ import annotations
import os
import re
import math
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union
import torch
import requests
from pydantic import ValidationError, BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from langgraph.graph import StateGraph, END
from pypdf import PdfReader, PdfWriter
from bs4 import BeautifulSoup
from config import *
#로그 파일 설정
current_path = os.getcwd()
log_filename = os.path.join(current_path, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
print(f"로그 파일이 다음 경로에 생성됩니다: {log_filename}")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True, # 보고서 별로 로그 파일 따로 기록
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # 파일 저장
        logging.StreamHandler(sys.stdout)                  # 터미널 동시 출력
    ]
)

#래퍼 함수
def print(*args, **kwargs):
    logging.info(" ".join(map(str, args)))

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

# 시간순 정렬
def sort_period_keys(keys: List[PeriodKey]) -> List[PeriodKey]:
    def kf(k: str) -> Tuple[int, int]:
        m = re.match(r"(\d{4})Q([1-4])", k)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0) #정규화된 형식(YYYYQn) 아닌 경우 (0,0)반환
    return sorted(keys, key=kf)

#직전 분기를 reference period로 추출
def pick_reference_index(periods: List[PeriodKey], idx_now: int, compare: CompareMode) -> Optional[int]:
    if idx_now < 0 or idx_now >= len(periods):
        return None
    m = re.match(r"(\d{4})Q([1-4])", periods[idx_now])
    if not m:
        return None
    y, q = int(m.group(1)), int(m.group(2))
    if compare == "YoY":
        target = f"{y-1}Q{q}"
    else:
        target = f"{y-1}Q4" if q == 1 else f"{y}Q{q-1}"
    try:
        idx = periods.index(target)
        return idx if idx < idx_now else None
    except ValueError:
        return None


#단위 통일 (백만원 기준)
def unit_multiplier_to_million(u: Unit) -> float:
    mapping = {"백만원": 1.0, "천원": 1e-3, "원": 1e-6, "천만원": 10.0, "억원": 100.0}
    return mapping.get(u, 1.0)

#날짜 추출
def get_quarter(month):
    month = int(month)
    if 1 <= month <= 3: return "Q1"
    elif 4 <= month <= 6: return "Q2"
    elif 7 <= month <= 9: return "Q3"
    elif 10 <= month <= 12: return "Q4"
    return ""

# 날짜 추출 (최신 Upstage 통합 HTML 완벽 대응)
def extract_financial_periods(doc_text_or_bundle):
    if isinstance(doc_text_or_bundle, dict):
        html_list = [t.get("html", "") for t in doc_text_or_bundle.get("tables", [])]
        file_name = doc_text_or_bundle.get("file", "")
        clean_text = " ".join([file_name] + html_list)
    else:
        clean_text = str(doc_text_or_bundle)

    clean_text = clean_text.replace("\\n", " ").replace("\\t", " ").replace("&nbsp;", " ")
    clean_text = re.sub(r'<[^>]+>', ' ', clean_text)  
    clean_text = re.sub(r'\s+', ' ', clean_text)      

    pattern_full = r"(제\s*\d+\s*기\s*(?:[1-4]분기|반기|사업연도)?\s*)?(20\d{2})\s*[\.년\-\/]\s*(\d{1,2})\s*[\.월\-\/]\s*(\d{1,2})\s*[일\.]?\s*(?:부터|~|-|∼|—| )?\s*(20\d{2})\s*[\.년\-\/]\s*(\d{1,2})\s*[\.월\-\/]\s*(\d{1,2})\s*[일\.]?(?:\s*까지)?"
    matches_full = re.findall(pattern_full, clean_text)

    if matches_full:
        extracted_dates = []
        for match in matches_full:
            term_str, start_y, start_m, start_d, end_y, end_m, end_d = match
            clean_term = term_str.strip() if term_str else ""
            end_date_str = f"{end_y}.{int(end_m):02d}.{int(end_d):02d}"
            quarter = get_quarter(end_m)
            period_key = f"{end_y}{quarter}"

            # 중복 방지
            if not any(d[0] == end_date_str for d in extracted_dates):
                extracted_dates.append((end_date_str, period_key, clean_term))

        # 내림차순 정렬 (최신 날짜가 맨 앞으로 오도록)
        extracted_dates.sort(reverse=True, key=lambda x: x[0])

        if len(extracted_dates) >= 2:
            return extracted_dates[0][0], extracted_dates[0][1], extracted_dates[0][2], extracted_dates[1][0], extracted_dates[1][1], extracted_dates[1][2]
        elif len(extracted_dates) == 1:
            return extracted_dates[0][0], extracted_dates[0][1], extracted_dates[0][2], "추출불가", f"{int(extracted_dates[0][1][:4])-1}{extracted_dates[0][1][4:]}", "전기"
#전체 보고서 중 포괄손익계산서 부분만 추출
def auto_slice_financials(input_path, output_path):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    target_pages = []
    print(f"문서 스캔 중... (총 {len(reader.pages)}페이지)")
    metric_keywords = [
        "매출", "매출액", "영업수익",
        "영업이익", "영업이익(손실)",
        "당기순이익", "당기순이익(손실)",
        "이자수익", "기타비용", "법인세비용", "법인세차감전순이익",
        "기타포괄손익", "배당수익", "보험손익", "보험비용", "투자손익"
    ]
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        clean_text = text.replace(" ", "").replace("\n", "")

        if "포괄손익계산서" in clean_text and "연결" not in clean_text:
            hit_count = sum(1 for k in metric_keywords if k in clean_text)
            if hit_count >= 2:
                print(f"(별도)포괄손익계산서 페이지 발견: {i}")
                target_pages.append(i)
  
    if not target_pages:
        print("전체 페이지 사용")
        for i in range(len(reader.pages)):
            writer.add_page(reader.pages[i])

        with open(output_path, "wb") as f:
            writer.write(f)
        return output_path

    # 첫 페이지 + 다음 페이지 추출 (표 최대 두 페이지에 작성된다고 생각)
    first = target_pages[0]
    pages = [first]
    if first + 1 < len(reader.pages):
        pages.append(first + 1)

    for p in pages:
        writer.add_page(reader.pages[p])

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"필터링 완료: {pages}")
    return output_path

#table html/text를 bundle로 정리해서 LLM에 넘김
class UpstageDocumentParseClient:
    def __init__(self):
        self.api_key = os.environ.get("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY is required.")
        self.base_url = "https://api.upstage.ai/v1/document-digitization"

    def parse(self, pdf_path: str, timeout: int = 600) -> Dict[str, Any]:
        url = self.base_url
        headers = {"Authorization": f"Bearer {self.api_key}"}

        data = {
            "model": "document-parse-260128",
            "ocr": "auto",
            "chart_recognition": "true",  # requests 폼 전송을 위해 문자열로 처리
            "coordinates": "true",
            "output_formats": '["html"]',
            "base64_encoding": '["figure"]',
        }

        with open(pdf_path, "rb") as f:
            files = {"document": f}
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

class LLMClient:
    def __init__(self, model_id: str, use_cpu: bool=False):
        self.device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{model_id} 로딩 중... device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def generate(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 모델이 혼자 사용자(<|user|>) 흉내를 내면 그 앞까지만 사용
        if "<|user|>" in content:
            content = content.split("<|user|>")[0].strip()
        if "<|assistant|>" in content:
            content = content.split("<|assistant|>")[0].strip()

        content = content.replace("```json", "").replace("```", "").strip()

        # 최종적으로 순수 JSON 괄호 { } 안의 내용만 추출
        start = content.find('{')
        end = content.rfind('}') + 1

        return content[start:end] if start != -1 and end > start else content

#LLM Table Extractor (HTML 파싱 없이 LLM로 표 읽기)
class MetricSelector:
    """Pass 1: 문서를 보고 분석할 핵심 지표 풀을 확정합니다."""
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, doc_bundle: Dict[str, Any], candidates: List[str]) -> List[str]:
        # MetricSelectionOutput에서 지정한 형태로 답하도록 요청
        # raw = '{"selected_metrics": ["매출", "이익"]}' 이런 식으로 답하게 됨
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
        user = (
            f"DOCUMENT_BUNDLE:\n{json.dumps(doc_bundle, ensure_ascii=False, indent=2)}\n\n"
            f"SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
        )
        raw = self.llm.generate(system=system, user=user)
        return MetricSelectionOutput(**json.loads(raw)).selected_metrics


class LLMTableExtractor:
    """Pass 2: Pass 1에서 확정된 지표에 대해서만 값을 추출합니다."""
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self,
        doc_bundle: Dict[str, Any],
        compare: CompareMode,
        selected_metrics: List[str]
    ) -> LLMExtractionOutput:
        schema = LLMExtractionOutput.model_json_schema()
        now_date, now_key, now_term, ref_date, ref_key, ref_term = extract_financial_periods(doc_bundle)

        system = (
"너는 제공된 문서에서 재무제표 핵심 수치만 선별해 정확한 수치를 추출하는 수석 재무 분석가다.\n"

"[분석 대상 기간]\n"
"아래 4개의 Key는 반드시 아래 지정해준 값으로 JSON에 넣어.\n"
f"- 'now_period_key': '{now_key}'\n"
f"- 'now_period': '{now_date}' (표에서 데이터를 찾을 때 참고할 열 이름표: '{now_term}')\n"
f"- 'ref_period_key': '{ref_key}'\n"
f"- 'ref_period': '{ref_date}' (표에서 데이터를 찾을 때 참고할 열 이름표: '{ref_term}')\n"

"[목표]\n"
"- 탐색 범위: 제공된 데이터의 `texts`(일반 텍스트 영역)와 `tables`(표 영역) 두 곳을 모두 탐색해라.\n"
"- 추출 대상: 위 두 영역을 종합하여 '문서에 기록된 원본 단위(unit_raw)', '당기기간 및 데이터', '비교기간 및 데이터'를 찾아 추출해라.\n"
"- 결과를 반드시 제공된 스키마와 일치하는 JSON으로만 출력해라.\n"

"[데이터 추출 절대 원칙: 원본 100% 일치]\n"
"1. 지표명(Key): 반드시 문서에 적힌 글자 그대로 출력해.\n"
"2. 수치(Value): 단 한 글자도 바꾸지 말고 표에 보이는 그대로 복사해서 출력해.\n"

"[지표명 복사 예시 (형식 변형 절대 금지)]\n"
"- 괄호() 없는 원본: 영업이익 (O) / 영업이익(손실) (X)\n"
"- 괄호() 있는 원본: 금융수익(손실) (O) / 금융수익 (X)\n"

"[수치 복사 예시 (형식 변형 절대 금지)]\n"
"- 쉼표(,) 유지: 489,341,873 (O) / 489341873 (X)\n"
"- 괄호() 없는 원본: 1234 (O) / (1234) (X)\n"
"- 괄호() 있는 원본: (1470) (O) / 1470 (X)\n"

"[값 추출 규칙]\n"
f"1. 반드시 {selected_metrics}에 포함된 지표들만 추출해.\n"
f"3. 당기({now_term}) 데이터는 지표명 우측에 있는 **첫 번째 숫자**('3개월' 열)를 추출해.\n"
f"4. 전기({ref_term}) 데이터는 지표명 우측에 있는 **세 번째 숫자**('3개월' 열)를 추출해.\n"
"5. 'items' 딕셔너리의 최상위 Key는 반드시 위에서 만든 PeriodKey여야 해.\n"
"6. 모든 수치 값은 반드시 문자열로 출력해. 숫자형으로 출력하지 마.\n"
"7. 모든 수치 반드시 맨 끝자리까지 정확하게 출력해.\n"

"[자체 검증 - 반드시 수행]\n"
"- 최종 출력은 반드시 스키마와 일치하는 JSON만 출력하라."
)
        user = (
            f"DOCUMENT_BUNDLE:\n{json.dumps(doc_bundle, ensure_ascii=False, indent=2)}\n\n"
            f"SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n"
            f"COMPARE_HINT: {compare}"
        )
        raw = self.llm.generate(system=system, user=user)

        return LLMExtractionOutput(**json.loads(raw))

class GrowthReasoner:
    def __init__(self, llm): 
        self.llm = llm

    def run(self, inp) -> GrowthReasoningOutput: 
        now_p = inp.now_period
        ref_p = inp.ref_period
        unit = inp.unit

        table_rows = [f"| 지표 | {ref_p} ({unit}) | {now_p} ({unit}) | 변동액 ({unit}) | 증감률 (%) |", "|---|---|---|---|---|"]
        analysis_hints = []

        for m in inp.top_moves:
            pct_str = f"{m.delta_pct*100:.1f}%" if m.delta_pct is not None else "n/a"
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

        user_content = f"실적 데이터:\n" + "\n".join(analysis_hints)
        raw = self.llm.generate(system=system, user=user_content)


        try:
            raw_text = raw.strip()
            if "```" in raw_text:
                raw_text = re.sub(r'```[a-zA-Z]*\n', '', raw_text)
                raw_text = raw_text.replace('```', '').strip()

            # [총평] 파싱
            growth_traj = "분석 요약 생성 실패"
            if "[총평]" in raw_text:
                traj_match = re.split(r'\[지표분석\]|\[특이사항\]', raw_text.split("[총평]")[1])[0]
                growth_traj = traj_match.strip()

            # [지표분석] 파싱
            analysis_list = []
            if "[지표분석]" in raw_text:
                ana_match = re.split(r'\[특이사항\]|\[총평\]', raw_text.split("[지표분석]")[1])[0]
                lines = ana_match.strip().split('\n')
                analysis_list = [line.strip().lstrip('-').lstrip('*').strip() for line in lines if line.strip()]

            if not analysis_list:
                analysis_list = [f"{h} 변동이 확인되었습니다." for h in analysis_hints]

            # [특이사항] 파싱
            caveats = []
            if "[특이사항]" in raw_text:
                cav_match = re.split(r'\[총평\]|\[지표분석\]', raw_text.split("[특이사항]")[1])[0]
                lines = cav_match.strip().split('\n')
                caveats = [line.strip().lstrip('-').lstrip('*').strip() for line in lines if line.strip()]

            return GrowthReasoningOutput(
                growth_trajectory=growth_traj,
                key_changes=analysis_list,
                kpi_summary=[], 
                summary_table=perfect_markdown_table,
                caveats=caveats
            )

        except Exception as e:
            print(f"[GrowthReasoner] 텍스트 파싱 에러 발생: {e}")
            return GrowthReasoningOutput(
                growth_trajectory="AI 분석 텍스트 파싱에 실패하여 수치 요약만 제공합니다.",
                key_changes=[f"{h} 변동이 확인됩니다." for h in analysis_hints],
                kpi_summary=[],
                summary_table=perfect_markdown_table,
                caveats=["AI 응답 형식이 깨져 원본 수치로 대체되었습니다."]
            )

#LangGraph 
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
      print("\n[Upstage 파싱 실패 상세 원인]")
      print(state.get("warnings"))
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

def run_pipeline(pdf_paths: List[str], llm: LLMClient, upstage: UpstageDocumentParseClient, compare: CompareMode = "YoY", top_k: int = 5, use_reasoning: bool = False):
    app = build_graph(MetricSelector(llm), LLMTableExtractor(llm), upstage, GrowthReasoner(llm) if use_reasoning else None)
    final_state = app.invoke({"pdf_paths": pdf_paths, "compare": compare, "top_k": top_k, "warnings": []})
    return final_state, render_report(final_state)

if __name__ == "__main__":
    upstage = UpstageDocumentParseClient()
    llm = LLMClient(model_id="kakaocorp/kanana-1.5-2.1b-instruct-2505")
    # 파일 경로 설정 
    original_pdf = "./[폴라리스오피스]분기보고서(2025.11.14).pdf"
    sliced_pdf = "./temp_sliced_finance.pdf" # 잘라낸 파일 임시 저장

    # 자동 스캔 및 자르기 실행
    print("PDF 자동 분리를 시작합니다...")
    auto_slice_financials(original_pdf, sliced_pdf)

    # 파이프라인 실행 
    report_state, text = run_pipeline(
        pdf_paths=[sliced_pdf], # 자른 파일을 투입
        llm=llm,
        upstage=upstage,
        compare="YoY",
        top_k=5,
        use_reasoning=True
    )

    # 결과 출력
    print("\n[LLM 추출 Raw Data]")
    print(json.dumps(report_state["llm_extraction_by_pdf"], indent=2, default=lambda x: x.__dict__, ensure_ascii=False))

    print("\n" + "="*60)
    print("\n[최종 생성된 분석 보고서]")
    print(text)
