# app/main.py
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.routes import router, rag_engine
from app.core.rag_engine import DocuMindRAG
from app.utils.config_loader import load_config

# 加载环境变量
load_dotenv()

# 配置日志（生产级可改为 JSON 格式）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("uvicorn")

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动时执行 ---
    logger.info("🚀 正在初始化 RAG 引擎...")
    config = load_config("config/config.json")  # 注意路径
    engine = DocuMindRAG(config)
    engine.build_index(force_rebuild=False)
    
    # 将引擎实例挂载到路由模块的全局变量中
    import app.api.routes as routes
    routes.rag_engine = engine
    logger.info("✅ RAG 引擎初始化完成")
    
    yield  # 服务运行中
    
    # --- 关闭时执行 ---
    logger.info("🛑 正在关闭 RAG 引擎...")
    # 可在此处持久化统计信息等

# 创建应用
app = FastAPI(
    title="DocuMind RAG API",
    description="基于 Rerank 的本地文档问答系统",
    version="0.1.0",
    lifespan=lifespan
)

# 跨域配置（允许前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "DocuMind is running"}