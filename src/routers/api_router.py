from fastapi import APIRouter
from ..models.models import QueryRequest, QueryResponse

# 创建路由器实例
router = APIRouter()

# 依赖项将在主应用中注入
router.query_service = None


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
        根据query进行向量检索
    """
    return await router.query_service(request)