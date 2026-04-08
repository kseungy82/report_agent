import os

# 환경 및 경로 설정
# 작업 폴더 경로
WORKSPACE_DIR = '/Users/kimseungyeon1/Desktop/금융sllm/'

# API 키 설정 
HUGGINGFACEHUB_API_TOKEN = "여기에_허깅페이스_토큰을_입력하세요"
UPSTAGE_API_KEY = "여기에_업스테이지_API_키를_입력하세요"

# 환경 변수 등록
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

# 모델 설정
MODEL_ID = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

# 재무 분석 지표 후보군
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
