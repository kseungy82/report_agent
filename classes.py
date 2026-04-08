import os
import json
import re
import torch
import requests
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import CompareMode, CANDIDATE_METRICS
from utils import (
    MetricSelectionOutput, LLMExtractionOutput, GrowthReasoningOutput, 
    GrowthReasoningInput, extract_financial_periods
)

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

        system = f"""
너는 제공된 문서에서 재무제표 핵심 수치만 선별해 정확한 수치를 추출하는 수석 재무 분석가다.

[분석 대상 기간]
아래 4개의 Key는 반드시 아래 지정해준 값으로 JSON에 넣어.
- "now_period_key": "{now_key}"
- "now_period": "{now_date}" (※ 표에서 데이터를 찾을 때 참고할 열 이름표: "{now_term}")
- "ref_period_key": "{ref_key}"
- "ref_period": "{ref_date}" (※ 표에서 데이터를 찾을 때 참고할 열 이름표: "{ref_term}")

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
