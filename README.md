# 语义FAQ搜索系统

## 功能特性

### 核心功能
- **智能向量化**: 使用 BAAI/bge-large-zh-v1.5 模型进行中文文本向量化
- **向量检索**: 基于 Milvus 向量数据库的高性能相似度搜索
- **多格式文档处理**: 支持 TXT、PDF、Word、Markdown 等多种文档格式
- **QA对生成**: 基于非结构化文档自动生成高质量问答对

### 高级功能
- **QA校验**: 自动校验生成的问答对质量，过滤不合格内容
- **智能润色**: 使用大语言模型对问答对进行内容优化和润色
- **批量测试**: 支持 recall@k 相似度检索测试和性能评估
- **性能监控**: 实时监控 GPU、CPU、内存使用情况并生成详细报表

### API服务
- **RESTful API**: 基于 FastAPI 的完整 Web API 接口
- **文件上传处理**: 支持批量文件上传和处理
- **异步处理**: 所有接口支持异步处理，提高并发性能

## 系统架构

```
semantic-faq-search/
├── data/                    # 数据存储目录
│   ├── .gitkeep
│   ├── processed_qa_pairs.csv
│   └── validated_data/      # 校验后的数据
├── docker/                  # Docker 部署配置
│   ├── docker-compose.yml
│   └── Dockerfile
├── outputs/                 # 输出文件目录
│   ├── polished_data/       # 润色后的数据
│   ├── test_reports/        # 测试报告
│   └── vectorization_reports/ # 向量化报告
├── src/                     # 核心源码目录
│   ├── core/                # 核心业务逻辑
│   │   ├── pipeline.py      # 数据处理管道
│   │   ├── document_processor.py # 文档处理
│   │   ├── qa_validator.py  # QA校验
│   │   └── data_tracer.py   # 数据追踪
│   ├── models/              # 数据模型定义
│   │   └── models.py
│   ├── repositories/        # 数据存储层
│   │   └── milvus_store.py  # Milvus 向量数据库
│   ├── routers/             # API 路由层
│   │   └── api_router.py    # REST API 路由
│   ├── services/            # 业务服务层
│   │   ├── service.py       # 主服务
│   │   ├── polish_service.py # 润色服务
│   │   └── test_services/   # 测试服务
│   ├── utils/               # 工具模块
│   │   ├── common.py        # 公共工具
│   │   ├── io_utils.py      # IO 工具
│   │   └── metrics.py       # 性能监控
│   └── prompts/             # 提示词模板
│       └── prompts.py
├── config.json              # 项目配置文件
├── requirements.txt         # Python 依赖清单
├── tool_configs/            # 工具配置
│   ├── milvus_config.json
│   └── test_config.json
└── README.md
```

## 环境要求

- Docker & Docker Compose
- NVIDIA GPU (支持CUDA 11.8)
- NVIDIA Container Toolkit

## 快速开始

### 1. 环境准备

确保已安装 Docker 和 Docker Compose，并且系统支持 NVIDIA GPU。

### 2. 克隆项目

```bash
git clone <repository-url>
cd semantic-faq-search
```

### 3. 启动基础服务

```bash
cd docker
docker-compose up -d
docker compose build service
```

### 4. 构建应用服务

```bash
# 数据向量化并且入库
docker-compose run app python3 -m src.core.pipeline
```

### 5. 运行数据处理管道

```bash
# 执行完整的数据处理流程：文档处理 -> QA生成 -> 校验 -> 润色 -> 向量化 -> 入库
docker-compose run --rm app python3 -m src.core.pipeline
```

### 6. 启动检索服务

```bash
docker-compose up service
```

服务将在 `http://localhost:8888` 启动。

## API 接口

服务启动后，提供以下 REST API 接口：

### 健康检查

```bash
curl http://localhost:8888/health
```

### 语义检索

```bash
curl -X POST http://localhost:8888/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何进行销售培训",
    "top_k": 10
  }'
```

**响应格式：**
```json
{
  "search_info": {
    "query": "如何进行销售培训",
    "timestamp": "2026-01-30T01:35:39.079000",
    "total_results": 25
  },
  "categories": [
    {
      "category_name": "销售培训",
      "items": [
        {
          "similarity_score": 95.2,
          "generate_source": "document_processor",
          "question": "如何进行有效的销售培训？",
          "answer": "销售培训需要结合理论和实践...",
          "image_url": ""
        }
      ]
    }
  ]
}
```

### 文档处理接口

#### 处理上传文件生成QA对

```bash
curl -X POST http://localhost:8888/process-uploaded-files \
  -F "files=@document.pdf" \
  -F "files=@manual.docx" \
  -F "service_name=销售培训" \
  -F "user_name=admin"
```

#### 文档处理+润色+向量化一体化

```bash
curl -X POST http://localhost:8888/process-document-with-polish \
  -F "file=@training_guide.pdf" \
  -F "service_name=销售培训" \
  -F "user_name=admin" \
  -F "drop_existing=false"
```

### 数据处理接口

#### CSV文件向量化

```bash
curl -X POST http://localhost:8888/vectorize-dataset-upload \
  -F "file=@qa_pairs.csv" \
  -F "drop_existing=false"
```

#### QA对校验

```bash
curl -X POST http://localhost:8888/validate-qa-pairs \
  -F "file=@qa_pairs.csv"
```

### 测试接口

#### 批量检索测试

```bash
curl -X POST http://localhost:8888/test/run-qa-test-upload \
  -F "file=@test_queries.csv" \
  -F "top_k=5" \
  -F "recall_k_values=1,3,5"
```

## 性能报表

执行流程后，在 `outputs/` 目录下生成性能报表：

- `performance_report_YYYYMMDD_HHMMSS.json`: JSON格式详细数据
- `performance_report_YYYYMMDD_HHMMSS.html`: 可视化HTML报表

报表包含各阶段的：
- 执行时长
- GPU利用率
- 内存使用情况
- 时间占比分析

## 配置说明

### 主要配置项

#### 数据处理配置

```json
{
  "data": {
    "input_csv": "data/input.csv",
    "text_columns": ["question", "answer"],
    "id_column": "ID"
  }
}
```

#### 文档处理配置

```json
{
  "document_processing": {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "max_qa_pairs_per_chunk": 5,
    "supported_extensions": [".txt", ".pdf", ".docx", ".md"],
    "enable_qa_validation": true
  }
}
```

#### 向量化配置

```json
{
  "embedding": {
    "model_name": "BAAI/bge-large-zh-v1.5",
    "batch_size": 128,
    "device": "cuda",
    "local_files_only": true
  }
}
```

#### Milvus 向量数据库配置

```json
{
  "milvus": {
    "host": "milvus-standalone",
    "port": 19530,
    "collection_name": "multilingual_vectors",
    "dimension": 1024,
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE"
  }
}
```

#### 服务配置

```json
{
  "service": {
    "host": "0.0.0.0",
    "port": 8888
  }
}
```

#### 润色服务配置

```json
{
  "polish": {
    "base_url": "http://192.168.151.84:8010",
    "timeout": 60,
    "batch_size": 50
  }
}
```

## 技术栈

### AI/ML 框架
- **向量化模型**: BAAI/bge-large-zh-v1.5 (基于 Sentence Transformers)
- **大语言模型**: Qwen3-max (通义千问)
- **深度学习**: PyTorch 2.x + CUDA 11.8

### 基础设施
- **向量数据库**: Milvus 2.3.3
- **Web框架**: FastAPI (异步高性能)
- **文档处理**: PyMuPDF, python-docx, python-pptx 等
- **监控工具**: psutil, nvidia-ml-py3

### 开发工具
- **容器化**: Docker + NVIDIA Container Toolkit
- **配置管理**: JSON 配置系统
- **日志系统**: Python logging
- **性能监控**: 自定义指标收集器

## 注意事项

### 环境要求
1. **GPU 支持**: 需要 NVIDIA GPU，支持 CUDA 11.8+
2. **内存**: 建议 16GB+ RAM，Milvus 需要约 2GB 内存
3. **存储**: 建议使用 SSD，模型和数据存储需要约 10GB+ 空间

### 模型下载
1. **首次运行**: 会自动下载 BAAI/bge-large-zh-v1.5 模型（约 1.2GB）
2. **网络**: 确保网络连接正常，模型托管在 Hugging Face
3. **缓存**: 模型会缓存在 `~/.cache/huggingface/` 目录

### 服务依赖
1. **Milvus**: 向量数据库服务，需要单独启动
2. **MinIO**: 对象存储服务，用于 Milvus 后端存储
3. **etcd**: Milvus 元数据存储

### 数据安全
1. **Collection 命名**: 默认使用 `multilingual_vectors`，请勿与其他项目冲突
2. **数据持久化**: 通过 Docker volumes 确保数据不会丢失

## 故障排除

### GPU未识别

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Milvus连接失败

检查服务状态：
```bash
docker-compose ps
docker-compose logs milvus-standalone
```

### 内存不足

调整配置中的 `batch_size` 参数降低内存占用。

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持多格式文档处理和 QA 对生成
- 集成向量检索和测试功能
- 完整的 Docker 容器化部署