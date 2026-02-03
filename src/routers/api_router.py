from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.models.models import QueryRequest, QueryResponse, ProcessFilesResponse, VectorizeDatasetResponse, ValidationResponse, ProcessDocumentWithPolishResponse, TestResponse, EnhanceAnswersResponse

# 创建路由器实例
router = APIRouter()

# 依赖项将在主应用中注入
router.query_service = None
router.process_uploaded_files_service = None
router.vectorize_dataset_upload_service = None
router.validate_qa_pairs_service = None
router.process_document_with_polish_service = None
router.run_qa_test_with_upload_service = None
router.enhance_answers_service = None

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


@router.post("/vectorize-dataset-upload", response_model=VectorizeDatasetResponse)
async def vectorize_dataset_upload(
    file: UploadFile = File(...),
    drop_existing: Optional[bool] = None
):
    """
        上传csv文件进行向量化存储
    """
    return await router.vectorize_dataset_upload_service(file, drop_existing)


@router.post("/validate-qa-pairs", response_model=ValidationResponse)
async def validate_qa_pairs(file: UploadFile = File(...)):
    """
        校验CSV文件中的question和answer字段，删除不合格样本
    """

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="仅支持CSV文件")

    return await router.validate_qa_pairs_service(file)


@router.post("/process-document-with-polish", response_model=ProcessDocumentWithPolishResponse)
async def process_document_with_polish(
    file: UploadFile = File(...),
    service_name: str = "",
    user_name: str = "",
    is_stream: bool = False,
    drop_existing: bool = False
):
    """
        处理文档生成润色后的QA对并存入向量数据库自动化
    """
    # 初步验证
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    return await router.process_document_with_polish_service(file, service_name, user_name, is_stream, drop_existing)


@router.post("/test/run-qa-test-upload", response_model=TestResponse)
async def run_qa_test_with_upload(
    file: UploadFile = File(...),
    top_k: Optional[int] = None,
    recall_k_values: Optional[str] = None
):
    """
        使用上传文件运行QA测试
    """
    # 初步验证
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 解析recall_k_values参数
    k_values = None
    if recall_k_values:
        try:
            k_values = [int(k.strip()) for k in recall_k_values.split(',')]
        except ValueError:
            raise HTTPException(status_code=400, detail="recall_k_values格式错误，应为逗号分隔的数字")

    return await router.run_qa_test_with_upload_service(file, top_k, k_values)


@router.post("/enhance-answers", response_model=EnhanceAnswersResponse)
async def enhance_answers(files: List[UploadFile] = File(...)):
    """
        批量增强答案内容
    """
    return await router.enhance_answers_service(files)


@router.get("/health")
async def health():
    """
        健康检查接口
    """
    return {"status": "healthy", "service": "multilingual-vector-search"}
