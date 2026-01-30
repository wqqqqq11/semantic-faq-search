from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import pandas as pd
import io
from datetime import datetime
from ..utils.common import load_config, setup_logger
from ..models.models import (
    CLIPEmbedder, QueryRequest, QueryResultItem, CategoryItem, QueryResponse,
    ProcessFilesResponse, VectorizeDatasetResponse, ValidationItem, ValidationResponse,
    ProcessDocumentWithPolishResponse
)
from ..repositories.milvus_store import MilvusStore
from ..core.document_processor import DocumentProcessor
from ..core.pipeline import Pipeline
from ..core.qa_validator import QAValidator
from .test_services.qas_test_service import QASTestService, TestRequest, TestResponse
from .polish_service import PolishService
from ..routers.api_router import router as api_router
from pymilvus import utility

app = FastAPI(title="多语种向量检索服务")

config = load_config()
logger = setup_logger("Service", config)
embedder = CLIPEmbedder(config)
store = MilvusStore(config)
document_processor = DocumentProcessor(config)
pipeline = Pipeline()
test_service = QASTestService()
qa_validator = QAValidator(config)
polish_service = PolishService(config)

# 设置路由器依赖项
api_router.embedder = embedder
api_router.store = store
api_router.logger = logger


async def query_service(request):
    """查询服务业务逻辑"""
    query_vector = embedder.encode([request.query])[0]

    results = store.search(
        query_vector=query_vector,
        top_k=min(request.top_k, 100)
    )

    # 按类别分组结果
    category_dict = {}
    for result in results:
        category = result.get('category', '未分类')
        if not category or category.strip() == '':
            category = '未分类'

        # 将相似度分数转换为百分比格式
        similarity_score = round(result['score'] * 100, 2)

        query_item = QueryResultItem(
            similarity_score=similarity_score,
            generate_source=result.get('generate_source', ''),
            question=result.get('question', ''),
            answer=result.get('answer', ''),
            image_url=result.get('image_url', '')
        )

        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append(query_item)

    # 转换为数组格式
    categories = []
    for category_name, items in category_dict.items():
        categories.append(CategoryItem(
            category_name=category_name,
            items=items
        ))

    # 构建搜索信息
    search_info = {
        "query": request.query,
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results)
    }

    return QueryResponse(
        search_info=search_info,
        categories=categories
    )


# 设置路由器服务
api_router.query_service = query_service

async def process_uploaded_files_service(files, service_name, user_name):
    """处理非结构化文档并生成并校验QA对数据集的业务逻辑"""
    # 准备文件数据
    uploaded_files = []
    for file in files:
        content = await file.read()
        uploaded_files.append({
            "name": file.filename,
            "content": content
        })

    # 处理文件
    qa_data = document_processor.process_uploaded_files(
        uploaded_files=uploaded_files,
        service_name=service_name,
        user_name=user_name
    )

    if not qa_data:
        return ProcessFilesResponse(
            success=False,
            message="未能生成任何QA对",
            qa_count=0,
            output_path=""
        )

    # 保存到CSV
    output_path = document_processor.save_to_csv(qa_data)

    return ProcessFilesResponse(
        success=True,
        message=f"成功处理 {len(files)} 个上传文件",
        qa_count=len(qa_data),
        output_path=output_path
    )

# 设置路由器服务
api_router.process_uploaded_files_service = process_uploaded_files_service



@app.on_event("startup")
async def startup():
    logger.info("启动服务")
    store.connect()
    # 确保集合存在（如果不存在则创建空集合）
    store.create_collection(drop_existing=False)
    store.load()
    logger.info("Milvus连接成功")


@app.on_event("shutdown")
async def shutdown():
    logger.info("关闭服务")
    store.disconnect()

# 注册路由器
app.include_router(api_router)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "multilingual-vector-search"}

@app.post("/test/run-qa-test-upload", response_model=TestResponse)
async def run_qa_test_with_upload(
    file: UploadFile = File(...),
    top_k: Optional[int] = None,
    recall_k_values: Optional[str] = None
):
    """使用上传文件运行QA测试"""
    try:
        # 解析recall_k_values参数
        k_values = None
        if recall_k_values:
            try:
                k_values = [int(k.strip()) for k in recall_k_values.split(',')]
            except ValueError:
                raise HTTPException(status_code=400, detail="recall_k_values格式错误，应为逗号分隔的数字")
        
        return await test_service.run_test_with_uploaded_file(file, top_k, k_values)
    except Exception as e:
        logger.error(f"上传文件QA测试错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def vectorize_dataset_upload_service(file, drop_existing):
    """上传csv文件进行向量化的业务逻辑"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="仅支持CSV文件")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = config.get('document_processing', {}).get('temp_dir', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"uploaded_dataset_{timestamp}.csv")

    content = await file.read()
    with open(temp_file_path, 'wb') as f:
        f.write(content)

    if drop_existing is None:
        drop_existing = config.get('vectorization', {}).get('drop_existing', False)

    result = pipeline.vectorize_dataset(temp_file_path, drop_existing)

    report_path = None
    if result.get('report'):
        report_dir = config.get('vectorization', {}).get('report_output_dir', 'outputs/vectorization_reports')
        timestamp_str = result['report'].get('timestamp', timestamp)
        report_path = os.path.join(report_dir, f"performance_report_{timestamp_str}.json")

    # 清理临时文件
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except:
            pass

    return VectorizeDatasetResponse(
        success=True,
        message=f"成功向量化 {result['total_records']} 条记录",
        total_records=result['total_records'],
        duration_seconds=round(result['duration_seconds'], 2),
        report_path=report_path
    )

# 设置路由器服务
api_router.vectorize_dataset_upload_service = vectorize_dataset_upload_service


async def validate_qa_pairs_service(file):
    """校验CSV文件中的question和answer字段的业务逻辑"""
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig')

    required_columns = ['question', 'answer']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"CSV文件缺少必要字段: {missing_columns}")

    id_column = 'ID' if 'ID' in df.columns else None
    results = []
    valid_indices = []
    valid_count = 0
    invalid_count = 0

    for idx, row in df.iterrows():
        question = str(row.get('question', '')).strip()
        answer = str(row.get('answer', '')).strip()
        row_id = row[id_column] if id_column else idx + 1

        is_valid, reason = qa_validator.validate_qa_pair(question, answer)

        if is_valid:
            valid_count += 1
            valid_indices.append(idx)
        else:
            invalid_count += 1

        results.append(ValidationItem(
            row_id=row_id,
            question=question,
            answer=answer,
            is_valid=is_valid,
            reason=reason
        ))

    total_count = len(results)
    pass_rate = (valid_count / total_count * 100) if total_count > 0 else 0

    output_path = None
    if valid_count > 0:
        valid_df = df.loc[valid_indices].copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('data', 'validated_data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"validated_qa_pairs_{timestamp}.csv")
        valid_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return ValidationResponse(
        success=True,
        message=f"校验完成，共 {total_count} 条记录，删除 {invalid_count} 条不合格记录",
        total_count=total_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        pass_rate=round(pass_rate, 2),
        output_path=output_path,
        results=results
    )

# 设置路由器服务
api_router.validate_qa_pairs_service = validate_qa_pairs_service


async def process_document_with_polish_service(file, service_name, user_name, is_stream, drop_existing):
    """处理文档生成润色后的QA对并存入向量数据库的业务逻辑"""
    uploaded_files = [{
        "name": file.filename,
        "content": await file.read()
    }]

    qa_data = document_processor.process_uploaded_files(
        uploaded_files=uploaded_files,
        service_name=service_name,
        user_name=user_name
    )

    if not qa_data:
        raise HTTPException(status_code=400, detail="未能生成任何QA对")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    validated_csv_path = os.path.join('data', f"validated_qa_{timestamp}.csv")
    os.makedirs('data', exist_ok=True)
    document_processor.save_to_csv(qa_data, validated_csv_path)

    polished_qa_data = await polish_service.polish_qa_pairs(qa_data)

    polished_csv_path = os.path.join('outputs', 'polished_data', f"polished_qa_{timestamp}.csv")
    os.makedirs(os.path.dirname(polished_csv_path), exist_ok=True)
    document_processor.save_to_csv(polished_qa_data, polished_csv_path)

    vectorization_result = pipeline.vectorize_dataset(polished_csv_path, drop_existing)

    return ProcessDocumentWithPolishResponse(
        success=True,
        message=f"处理完成，润色后生成 {len(polished_qa_data)} 条QA对",
        original_qa_count=len(qa_data),
        polished_qa_count=len(polished_qa_data),
        vectorized_count=vectorization_result['total_records'],
        validated_csv_path=validated_csv_path,
        polished_csv_path=polished_csv_path,
        vectorization_report=vectorization_result.get('report')
    )

# 设置路由器服务
api_router.process_document_with_polish_service = process_document_with_polish_service


def start_service():
    uvicorn.run(
        app,
        host=config['service']['host'],
        port=config['service']['port'],
        log_level="info"
    )


if __name__ == "__main__":
    start_service()
