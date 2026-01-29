from fastapi import APIRouter, UploadFile, File
from typing import List
from ..models.models import QueryRequest, QueryResponse, ProcessFilesResponse

# 创建路由器实例
router = APIRouter()

# 依赖项将在主应用中注入
router.query_service = None
router.process_uploaded_files_service = None

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
        根据query进行向量检索
    """
    return await router.query_service(request)


@router.post("/process-uploaded-files", response_model=ProcessFilesResponse)
async def process_uploaded_files(
    files: List[UploadFile] = File(...),
    service_name: str = "",
    user_name: str = ""
):
    """
        处理上传的非结构化文档并生成并校验QA对数据集
    """
    return await router.process_uploaded_files_service(files, service_name, user_name)