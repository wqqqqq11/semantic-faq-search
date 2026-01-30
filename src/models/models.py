from typing import Dict, List, Any, Optional

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from src.utils.common import retry


class CLIPEmbedder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config['clip']
        self.device = self.config['device'] if torch.cuda.is_available() else 'cpu'
        local_files_only = self.config.get('local_files_only', False)
        self.model = SentenceTransformer(
            self.config['model_name'], 
            device=self.device,
            local_files_only=local_files_only
        )
        self.batch_size = self.config['batch_size']
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """ 样本向量化"""

        if not texts:
            return np.array([])
        
        return self.model.encode(
            texts, 
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device
        )
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()



# API 模型定义
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResultItem(BaseModel):
    similarity_score: float
    generate_source: Optional[str] = ""
    question: str
    answer: str
    image_url: str


class CategoryItem(BaseModel):
    category_name: str
    items: List[QueryResultItem]


class QueryResponse(BaseModel):
    search_info: Dict[str, Any]
    categories: List[CategoryItem]


class ProcessFilesRequest(BaseModel):
    file_paths: List[str]
    service_name: Optional[str] = ""
    user_name: Optional[str] = ""
    output_path: Optional[str] = None


class ProcessFilesResponse(BaseModel):
    success: bool
    message: str
    qa_count: int
    output_path: str


class ProcessUploadedFilesRequest(BaseModel):
    service_name: str = ""
    user_name: str = ""


class VectorizeDatasetResponse(BaseModel):
    success: bool
    message: str
    total_records: int
    duration_seconds: float
    report_path: Optional[str] = None


class ValidationItem(BaseModel):
    row_id: Any
    question: str
    answer: str
    is_valid: bool
    reason: str


class ValidationResponse(BaseModel):
    success: bool
    message: str
    total_count: int
    valid_count: int
    invalid_count: int
    pass_rate: float
    output_path: Optional[str] = None
    results: List[ValidationItem]


class ProcessDocumentWithPolishResponse(BaseModel):
    success: bool
    message: str
    original_qa_count: int
    polished_qa_count: int
    vectorized_count: int
    validated_csv_path: str
    polished_csv_path: str
    vectorization_report: Optional[Dict[str, Any]] = None


class TestRequest(BaseModel):
    test_csv_path: Optional[str] = None
    top_k: Optional[int] = None
    recall_k_values: Optional[List[int]] = None


class TestMetrics(BaseModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    total_queries: int
    exact_matches: int


class TestResponse(BaseModel):
    success: bool
    message: str
    metrics: TestMetrics
    report_path: str
    timestamp: str
