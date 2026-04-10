# 금융 재무제표 분석 에이전트 (Financial Report Analysis Agent)

기업의 분기/사업보고서(PDF)를 입력하면 자동으로 `포괄손익계산서` 부분을 찾아내고, Upstage OCR과 로컬 sLLM을 활용해 핵심 재무 지표를 추출 및 분석해 주는 자동화 파이프라인입니다.

## 주요 기능
1. **PDF 자동 스캔 및 분리**: 수백 페이지의 보고서 중 `포괄손익계산서` 페이지(1~2장)만 초고속으로 탐색 및 추출합니다.
2. **HTML 기반 표 구조화**: Upstage Document Parse API를 이용해 복잡한 재무제표 표를 정확하게 읽어냅니다.
3. **핵심 지표 자율 선정 및 추출**: LangGraph 기반의 AI 에이전트가 기업의 특성에 맞는 핵심 지표(5~10개)를 스스로 선정하고 핵심 지표값을 추출합니다.
4. **재무 추이 요약 보고서 생성**: 추출된 데이터를 바탕으로 연도별 혹은 분기별 증감률을 계산하고, 표 형태로 지표 변화를 요약한 뒤 금융 전문가 어조의 서술형 분석 보고서를 작성합니다.

## 프로젝트 구조

- `main.py`: FastAPI 엔트리포인트
- `nodes.py`: 파이프라인/노드/LLM 호출 로직
- `utils.py`: 유틸 함수
- `classes.py`: 스키마/타입
- `router.py`: 상위 라우터
- `config.py`: `.env` 로딩

---

## 사전 준비 (Prerequisites)

이 프로젝트는 로컬 GPU 환경에서의 구동을 권장합니다.

1. **Python 3.10+** 권장 (`requirements.txt` 기준)
2. API 키 발급 필요
   - [Hugging Face Token](https://huggingface.co/) (모델 다운로드)
   - [Upstage API Key](https://upstage.ai/) (문서 OCR/파싱)

---

## 설치 및 환경 설정 (Installation & Setup)

### 1) 저장소 클론 및 폴더 이동

```bash
git clone https://github.com/kseungy82/report_agent.git
cd report_agent
```

### 2) 의존성 설치

```bash
python -m pip install -r requirements.txt
```

### 3) 환경변수 설정 (`.env`)

프로젝트 루트에 `.env` 파일 생성:

```env
UPSTAGE_API_KEY=up_xxxxx
HUGGINGFACEHUB_API_TOKEN=hf_xxxxx
KANANA_MODEL_ID=kakaocorp/kanana-1.5-2.1b-instruct-2505
APP_PORT=8000
LOG_DIR=.
RUNTIME_DIR=runtime_files
```

### 4) 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port ${APP_PORT:-8000} --log-level debug
```

또는:

```bash
python main.py
```

## 헬스체크

```bash
curl http://127.0.0.1:8000/health
```

정상 응답:

```json
{"ok": true}
```

## API 엔드포인트

- `GET /health`
  - 서버 상태 확인
  - 응답: `{"ok": true}`

- `POST /route`
  - **기본 엔드포인트** (상위 라우터 -> 하위 분석 에이전트 실행)
  - `multipart/form-data`
  - 필수: `task` (문자열), `pdf` (파일)
  - 선택:
    - `compare` (`YoY` | `QoQ`, 기본 `YoY`)
    - `top_k` (기본 `5`)
    - `use_reasoning` (기본 `true`)
    - `slice_financial_statement` (기본 `true`)

- `POST /analyze`
  - 보조 엔드포인트(디버깅/직접 실행용)
  - `multipart/form-data`
  - 필수: `pdf` (파일)
  - 선택:
    - `compare` (`YoY` | `QoQ`, 기본 `YoY`)
    - `top_k` (기본 `5`)
    - `use_reasoning` (기본 `true`)
    - `slice_financial_statement` (기본 `true`)

## 분석 요청

### (A) 기본 요청 - `/route` (권장)

```bash
curl -X POST "http://127.0.0.1:8000/route" \
  -F "task=분기보고서 핵심 지표 분석해줘" \
  -F "pdf=@/절대경로/파일.pdf"
```

### (B) 보조 요청 - `/analyze` (직접 실행)

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "pdf=@/절대경로/파일.pdf"
```

## 로그/파일 저장 위치

- 로그 파일: `LOG_DIR` 기준 `analysis_log_YYYYMMDD_HHMMSS.log` (기본 `.`)
- 업로드/슬라이스 파일: `RUNTIME_DIR` 디렉토리 (기본 `runtime_files/`)
