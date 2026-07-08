# app/api/routes.py
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import QueryRequest, QueryResponse, IndexRequest, IndexResponse
from app.core.rag_engine import DocuMindRAG
from typing import Dict, Any

router = APIRouter(prefix="/api/v1", tags=["RAG"])

# 全局单例（应用启动时初始化）
rag_engine: DocuMindRAG = None

def get_engine() -> DocuMindRAG:
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG 引擎尚未初始化")
    return rag_engine

@router.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest, engine: DocuMindRAG = Depends(get_engine)):
    """核心问答接口"""
    try:
        result = engine.query(req.question)
        # 返回时附带当前统计信息（可选）
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            stats=engine.stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

@router.post("/index", response_model=IndexResponse)
async def handle_index(req: IndexRequest, engine: DocuMindRAG = Depends(get_engine)):
    """重建向量索引（耗时操作）"""
    try:
        # 注意：重建索引可能耗时较长，生产环境应使用 Celery 异步任务。
        # 此处为了演示，直接同步执行，但通过 FastAPI 的 `def` 而非 `async def` 
        # 可以避免阻塞事件循环（FastAPI 会自动将其放到线程池）。
        engine.reset_index(force_rebuild=req.force) if req.force else engine.build_index(force_rebuild=False)
        count = engine.vector_store._collection.count() if engine.vector_store else 0
        return IndexResponse(status="success", message="索引已更新", vector_count=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引失败: {str(e)}")

@router.get("/stats")
async def handle_stats(engine: DocuMindRAG = Depends(get_engine)):
    """获取运行时统计"""
    return engine.stats