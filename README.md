# LlamaIndex 多索引混合检索系统

基于 LlamaIndex 构建的智能文档检索系统，支持多种索引类型和混合检索策略。

## 功能特性

- **多种索引类型**：
  - 向量索引（Pinecone）：语义相似性搜索
  - 摘要索引（MongoDB）：文档摘要和概述（使用 tree_summarize 模式）
  - 树索引（MongoDB）：层次化查询
  - 关键词索引（MongoDB）：关键词精确匹配
  - 属性图索引（Neo4j）：实体关系查询

- **混合检索**：
  - 自动选择最合适的检索策略
  - 综合多个索引的结果
  - 提高检索准确性和效率

- **API 速率限制**：
  - 防止超出 Google GenAI API 配额
  - 支持自定义请求间隔
  - 支持自定义 embedding 维度（768 维）

- **Langfuse 追踪**：
  - 自动追踪所有 LLM 调用
  - 监控性能和成本
  - 可视化调用链路

## 项目结构

```
try_llamaindex/
├── book/                    # 文档目录
│   └── 三国演义.txt
├── models.py                # 模型配置（LLM 和 Embedding，带速率限制）
├── storage.py               # 存储管理器（按存储后端组织）
├── process_document.py      # 文档处理和索引创建
├── hybrid_retrieval.py     # 混合检索
├── config.yaml            # 配置文件（提交到 Git）
├── .env                  # 环境变量（不提交到 Git）
├── .env.sample           # 环境变量示例
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明
```

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd try_llamaindex
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制示例配置文件并填入你的 API 密钥：

```bash
cp .env.sample .env
```

编辑 `.env` 文件，填入以下信息：

- `GOOGLE_API_KEY`: Google GenAI API 密钥
- `LANGFUSE_SECRET_KEY`: Langfuse 密钥（可选）
- `LANGFUSE_PUBLIC_KEY`: Langfuse 公钥（可选）
- `LANGFUSE_BASE_URL`: Langfuse 服务地址（可选）
- `PINECONE_API_KEY`: Pinecone API 密钥
- `MONGO_URI`: MongoDB 连接字符串
- `NEO4J_URI`: Neo4j 连接 URL
- `NEO4J_USERNAME`: Neo4j 用户名
- `NEO4J_PASSWORD`: Neo4j 密码
- `NEO4J_DATABASE`: Neo4j 数据库名

`config.yaml` 包含配置信息，根据情况修改。

## 使用方法

### 1. 创建索引

将文档放入 `book/` 目录，然后运行：

```bash
python process_document.py
```

这将：
- 加载文档
- 进行文档分块
- 创建向量索引、摘要索引、树索引、关键词索引、属性图索引
- 将索引直接存储到相应的云端存储（Pinecone、MongoDB、Neo4j）
- 使用 Langfuse 追踪所有 LLM 调用

**注意：**
- 属性图索引创建失败时会自动跳过，不影响其他索引
- 摘要索引使用 `tree_summarize` 模式生成高质量摘要
- 所有索引直接存储到云端，不保存到本地

### 2. 混合检索

运行混合检索程序：

```bash
python hybrid_retrieval.py
```

这将：
- 加载所有索引
- 创建混合查询引擎
- 根据问题自动选择最合适的检索策略
- 综合多个索引的结果生成答案

### 3. 自定义查询

在 `hybrid_retrieval.py` 中修改 `questions` 列表，或直接使用代码：

```python
from hybrid_retrieval import create_hybrid_query_engine, query_hybrid

query_engine = create_hybrid_query_engine()
response = query_hybrid(query_engine, "你的问题")
print(response)
```

## 配置说明

### Embedding 维度

默认使用 768 维 embedding，可在 `models.py` 中修改：

```python
embed_model = RateLimitedGoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embed_batch_size=1,
    request_delay=2.0,
    output_dimensionality=768  # 修改此处
)
```

### API 速率限制

默认每次请求间隔：
- LLM：1 秒
- Embedding：2 秒

可在 `models.py` 中修改：

```python
llm = RateLimitedGoogleGenAI(
    "gemini-3-flash-preview",
    request_delay=1.0  # 修改此处（秒）
)

embed_model = RateLimitedGoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embed_batch_size=1,
    request_delay=2.0,  # 修改此处（秒）
    output_dimensionality=768
)
```

### 文档分块参数

可在 `process_document.py` 中修改：

```python
nodes = chunk_documents(documents, chunk_size=1024, chunk_overlap=20)
```

### 摘要索引模式

摘要索引使用 `tree_summarize` 模式生成高质量层次化摘要：

```python
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True
)
```

## 检索策略

### 向量索引
- 适用场景：语义相似性搜索
- 优势：理解语义，找到相关内容
- 存储后端：Pinecone

### 摘要索引
- 适用场景：获取文档整体摘要
- 优势：使用 tree_summarize 模式生成高质量层次化摘要
- 存储后端：MongoDB

### 树索引
- 适用场景：层次化查询
- 优势：快速定位到相关章节
- 存储后端：MongoDB

### 关键词索引
- 适用场景：关键词精确匹配
- 优势：精确匹配关键词
- 存储后端：MongoDB

### 属性图索引
- 适用场景：实体关系查询
- 优势：理解实体间的关系
- 存储后端：Neo4j

## 存储架构

### 配置分离

```
配置文件：
├── .env              # 敏感信息（不提交到 Git）
│   ├── GOOGLE_API_KEY
│   ├── PINECONE_API_KEY
│   ├── MONGO_URI
│   ├── NEO4J_URI
│   ├── NEO4J_USERNAME
│   ├── NEO4J_PASSWORD
│   └── NEO4J_DATABASE
│
└── config.yaml       # 配置信息（提交到 Git）
    ├── pinecone: index_name, dimension, metric
    ├── mongodb: db_name, collection_name, namespace
    └── neo4j: database, index_name, text_node_property
```

### 存储后端

| 索引类型 | 存储后端 | 连接信息来源 | 配置信息来源 |
|-----------|-----------|--------------|--------------|
| 向量索引 | Pinecone | .env (PINECONE_API_KEY) | config.yaml |
| 摘要索引 | MongoDB | .env (MONGO_URI) | config.yaml |
| 树索引 | MongoDB | .env (MONGO_URI) | config.yaml |
| 关键词索引 | MongoDB | .env (MONGO_URI) | config.yaml |
| 属性图索引 | Neo4j | .env (NEO4J_*) | config.yaml |

## 注意事项

1. **API 配额**：Google GenAI API 有配额限制，建议使用速率限制功能
2. **存储成本**：Pinecone 和 Neo4j 可能有存储成本，注意监控使用量
3. **索引创建时间**：首次创建索引可能需要较长时间，取决于文档大小
4. **安全性**：不要将 `.env` 提交到 Git（`config.yaml` 已安全，可以提交）
5. **Neo4j 连接**：新创建的 Neo4j 实例可能需要等待 60 秒才能连接
6. **API 可用性**：Google GenAI API 可能出现 503 错误（服务不可用），建议稍后重试

## 故障排除

### ModuleNotFoundError

确保已安装所有依赖：

```bash
pip install -r requirements.txt
```

### API 配额超限

增加 `request_delay` 参数的值：

```python
request_delay=2.0  # 增加到 2 秒
```

### 连接失败

检查 `.env` 中的配置是否正确：

```bash
python test_neo4j.py  # 测试 Neo4j 连接
```

### Neo4j DatabaseNotFound 错误

确保 `config.yaml` 中的数据库名称与 Neo4j 实例一致：

```yaml
neo4j:
  database: "75565d13"  # 必须与 Neo4j 实例 ID 一致
```

### Google API 503 错误

```
google.genai.errors.ServerError: 503 UNAVAILABLE
```

**原因**：Google GenAI API 服务暂时不可用（负载过高）

**解决方案**：
- 等待 10-30 分钟后重试
- 或避开高峰时段

### 属性图索引创建失败

属性图索引创建失败时会自动跳过，不影响其他索引。错误信息会显示在终端中。

## Langfuse 追踪

系统自动追踪所有 LLM 调用，包括：

- 文档处理
- 索引创建
- 摘要生成
- 关键词提取
- 实体关系提取

可以在 Langfuse Dashboard 中查看：
- 性能指标（延迟、吞吐量）
- 成本追踪（token 使用量）
- 错误追踪
- 调用链路可视化

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
