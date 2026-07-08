# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户提问内容", min_length=1)

class QueryResponse(BaseModel):
    question: str
    answer: str
    stats: Optional[Dict[str, Any]] = None

class IndexRequest(BaseModel):
    force: bool = Field(False, description="是否强制重建")

class IndexResponse(BaseModel):
    status: str
    message: str
    vector_count: Optional[int] = None