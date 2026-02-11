from typing import Dict, List, Any, Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel


class CLIPEmbedder:
    def __init__(self, config: Dict[str, Any]):
        cfg = config['clip']
        self._embeddings = HuggingFaceEmbeddings(
            model_name=cfg['model_name'],
            model_kwargs={
                "device": cfg.get('device', 'cpu'),
                "local_files_only": cfg.get('local_files_only', False),
            },
            encode_kwargs={"batch_size": cfg.get('batch_size', 128)},
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        vecs = self._embeddings.embed_documents(texts)
        return np.array(vecs, dtype=np.float32)

    def get_dimension(self) -> int:
        if not hasattr(self, "_dim"):
            self._dim = len(self._embeddings.embed_query("x"))
        return self._dim



# API 模型定义
class QueryRequest(BaseModel):
    """
    向量相似度检索请求
    """
    query: str
    top_k: int = 10


class QueryResultItem(BaseModel):
    """
    向量相似度检索结果项
    """
    similarity_score: float
    generate_source: Optional[str] = ""
    question: str
    answer: str
    image_url: str


class CategoryItem(BaseModel):
    """
    类别项
    """
    category_name: str
    items: List[QueryResultItem]


class QueryResponse(BaseModel):
    """
    向量相似度检索响应
    """
    search_info: Dict[str, Any]
    categories: List[CategoryItem]


class ProcessFilesRequest(BaseModel):
    """
    处理上传文件请求
    """
    file_paths: List[str]
    service_name: Optional[str] = ""
    user_name: Optional[str] = ""
    output_path: Optional[str] = None


class ProcessFilesResponse(BaseModel):
    """
    处理上传文件响应
    """
    success: bool
    message: str
    qa_count: int
    output_path: str


class ProcessUploadedFilesRequest(BaseModel):
    """
    处理上传文件请求
    """
    service_name: str = ""
    user_name: str = ""


class VectorizeDatasetResponse(BaseModel):
    """
    向量化数据集响应
    """
    success: bool
    message: str
    total_records: int
    duration_seconds: float
    report_path: Optional[str] = None


class ValidationItem(BaseModel):
    """
    校验项
    """
    row_id: Any
    question: str
    answer: str
    is_valid: bool
    reason: str


class ValidationResponse(BaseModel):
    """
    校验响应
    """
    success: bool
    message: str
    total_count: int
    valid_count: int
    invalid_count: int
    pass_rate: float
    output_path: Optional[str] = None
    results: List[ValidationItem]


class ProcessDocumentWithPolishResponse(BaseModel):
    """
    处理文档并润色响应
    """
    success: bool
    message: str
    original_qa_count: int
    polished_qa_count: int
    vectorized_count: int
    validated_csv_path: str
    polished_csv_path: str
    vectorization_report: Optional[Dict[str, Any]] = None


class TestRequest(BaseModel):
    """
    测试请求
    """
    test_csv_path: Optional[str] = None
    top_k: Optional[int] = None
    recall_k_values: Optional[List[int]] = None


class TestMetrics(BaseModel):
    """
    测试指标
    """
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    total_queries: int
    exact_matches: int


class TestResponse(BaseModel):
    """
    测试响应
    """
    success: bool
    message: str
    metrics: TestMetrics
    report_path: str
    timestamp: str


class EnhanceRequestItem(BaseModel):
    question: str
    answer: str


class EnhanceResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]


class EnhanceAnswersResponse(BaseModel):
    success: bool
    message: str
    output_path: str
    total_processed: int
