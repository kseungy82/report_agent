# 금융 재무제표 분석 에이전트 (Financial Report Analysis Agent)

기업의 분기/사업보고서(PDF)를 입력하면 자동으로 '포괄손익계산서' 부분을 찾아내고, Upstage OCR과 로컬 sLLM을 활용해 핵심 재무 지표를 추출 및 분석해 주는 자동화 파이프라인입니다.

## 주요 기능
1. **PDF 자동 스캔 및 분리**: 수백 페이지의 보고서 중 '포괄손익계산서' 페이지(1~2장)만 초고속으로 탐색 및 추출합니다.
2. **HTML 기반 표 구조화**: Upstage Document Parse API를 이용해 복잡한 재무제표 표를 정확하게 읽어냅니다.
3. **핵심 지표 자율 선정 및 추출**: LangGraph 기반의 AI 에이전트가 기업의 특성에 맞는 핵심 지표(5~10개)를 스스로 선정하고 핵심 지표값을 추출합니다.
4. **재무 추이 요약 보고서 생성**: 추출된 데이터를 바탕으로 연도별 혹은 분기별 증감률을 계산하고, 표 형태로 지표 변화를 요약한 뒤, 금융 전문가의 어조로 실적 변동에 대한 서술형 분석 보고서를 작성합니다.

---

## 사전 준비 (Prerequisites)

이 프로젝트는 로컬 GPU에서의 구동을 최적화하여 설계되었습니다.

1. **Python 3.8+** 환경을 권장합니다.
2. API 키 발급이 필요합니다.
   - [Hugging Face Token](https://huggingface.co/) (모델 가중치 다운로드 용도)
   - [Upstage API Key](https://upstage.ai/) (문서 OCR 파싱 용도)

---

## 설치 및 환경 설정 (Installation & Setup)

**1. 저장소 클론 및 폴더 이동**
```bash
git clone https://github.com/kseungy82/report_agent.git
cd report_agent
