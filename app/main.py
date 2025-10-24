# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager

# 분리된 코어 모델 로드 함수 및 라우터 임포트
from app.core.models import load_translation_models, global_resources
from app.routers.translate import router as translation_router 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 (Startup): 모델 로드 함수 호출
    load_translation_models()
    
    yield # API가 실행되는 동안 대기
    
    # 서버 종료 시 (Shutdown): 정리
    print("모델 및 토크나이저 정리...")
    global_resources.clear() # global_resources 딕셔너리를 비워 리소스 정리
    print("정리 완료.")

# FastAPI 인스턴스 생성 및 lifespan 함수 적용
app = FastAPI(
    title="M2M100 Translation API (Fully Modular)",
    description="완전히 모듈화된 M2M100 다국어 번역 서비스 API",
    version="1.0.2",
    lifespan=lifespan
)

# 분리된 라우터 등록
# 최종 경로는 POST /api/v1/translate/ 가 됩니다.
app.include_router(translation_router, prefix="/api")


@app.get("/")
def read_root():
    """API 상태 확인을 위한 기본 엔드포인트"""
    return {"message": "M2M100 Translation API is running. Check /api/v1/translate/ for the main endpoint. Go to /docs for details."}