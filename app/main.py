from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 도메인 접근 허용
    allow_credentials=True,     # 쿠키 허용
    allow_methods=["*"],        # HTTP 메서드 허용
    allow_headers=["*"],        # 요청 허용
)

app.include_router(router)
