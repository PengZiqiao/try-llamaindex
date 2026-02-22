import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, TreeIndex, KeywordTableIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.response_synthesizers import get_response_synthesizer
from storage import StorageManager
from models import llm, embed_model
from langfuse import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# 允许在已有事件循环环境中嵌套运行异步代码（如 Jupyter Notebook）
nest_asyncio.apply()

# 初始化 Langfuse 客户端，用于追踪和监控 LLM 调用链路
langfuse = get_client()

# 初始化 LlamaIndex 的自动埋点工具，用于收集框架内部事件
instrumentor = LlamaIndexInstrumentor()
instrumentor.instrument()

storage_manager = StorageManager()


def load_documents(file_path):
    """
    加载文档
    
    Args:
        file_path (str): 文档路径
    
    Returns:
        list: 文档列表
    """
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    print(f"成功加载 {len(documents)} 个文档")
    return documents


def chunk_documents(documents, chunk_size=1024, chunk_overlap=20):
    """
    文档分块
    
    Args:
        documents (list): 文档列表
        chunk_size (int): 每个块的字符数，默认为 1024
        chunk_overlap (int): 块之间的重叠字符数，默认为 20
    
    Returns:
        list: 分块后的节点列表
    """
    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    print(f"文档分块完成，共生成 {len(nodes)} 个节点")
    return nodes[:2]


def create_vector_index(nodes):
    """
    创建向量索引并存储到云端
    
    使用 SummaryExtractor 为每个节点生成摘要
    
    Args:
        nodes (list): 已分块的节点列表
    
    Returns:
        VectorStoreIndex: 向量索引实例
    """
    print("\n=== 创建向量索引 ===")
    
    summary_extractor = SummaryExtractor(llm=llm, summaries=["self"])
    pipeline = IngestionPipeline(transformations=[summary_extractor])
    nodes_with_summary = pipeline.run(nodes=nodes, show_progress=True)
    
    cloud_storage_context = storage_manager.get_pinecone_storage_context()
    
    index = VectorStoreIndex(
        nodes=nodes_with_summary,
        storage_context=cloud_storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("向量索引已保存到 Pinecone")
    return index


def create_summary_index(nodes):
    """
    创建文档摘要索引并存储到云端
    
    DocumentSummaryIndex 会在创建时为每个文档生成摘要，并构建层次化的摘要结构
    使用 tree_summarize 模式生成高质量的层次化摘要
    
    Args:
        nodes (list): 已分块的节点列表
    
    Returns:
        DocumentSummaryIndex: 文档摘要索引实例
    """
    print("\n=== 创建摘要索引 ===")
    
    cloud_storage_context = storage_manager.get_mongodb_storage_context(namespace="summary_index")
    
    # 使用树形摘要模式异步生成高质量层次化摘要
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        use_async=True
    )
    
    index = DocumentSummaryIndex(
        nodes=nodes,
        storage_context=cloud_storage_context,
        llm=llm,
        embed_model=embed_model,
        response_synthesizer=response_synthesizer,
        show_progress=True
    )
    
    print("摘要索引已保存到 MongoDB")
    return index


def create_tree_index(nodes):
    """
    创建树索引并存储到云端
    
    Args:
        nodes (list): 已分块的节点列表
    
    Returns:
        TreeIndex: 树索引实例
    """
    print("\n=== 创建树索引 ===")
    
    cloud_storage_context = storage_manager.get_mongodb_storage_context(namespace="tree_index")
    
    index = TreeIndex(
        nodes=nodes,
        storage_context=cloud_storage_context,
        llm=llm,
        show_progress=True
    )
    
    print("树索引已保存到 MongoDB")
    return index


def create_keyword_index(nodes):
    """
    创建关键词索引并存储到云端
    
    Args:
        nodes (list): 已分块的节点列表
    
    Returns:
        KeywordTableIndex: 关键词索引实例
    """
    print("\n=== 创建关键词索引 ===")
    
    cloud_storage_context = storage_manager.get_mongodb_storage_context(namespace="keyword_index")
    
    index = KeywordTableIndex(
        nodes=nodes,
        storage_context=cloud_storage_context,
        llm=llm,
        show_progress=True
    )
    
    print("关键词索引已保存到 MongoDB")
    return index


def create_property_graph_index(nodes):
    """
    创建属性图索引并存储到云端
    
    Args:
        nodes (list): 已分块的节点列表
    
    Returns:
        PropertyGraphIndex: 属性图索引实例
    """
    print("\n=== 创建属性图索引 ===")
    
    kg_extractors = [
        SchemaLLMPathExtractor(llm=llm)
    ]
    
    graph_store = storage_manager.get_neo4j_property_graph_store()
    cloud_storage_context = storage_manager.get_neo4j_storage_context(embed_dim=768)
    
    index = PropertyGraphIndex(
        nodes=nodes,
        storage_context=cloud_storage_context,
        llm=llm,
        embed_model=embed_model,
        kg_extractors=kg_extractors,
        property_graph_store=graph_store,
        show_progress=True
    )
    
    print("属性图索引已保存到 Neo4j")
    return index


def main():
    """
    主函数：加载文档、分块、创建各种索引并存储到云端
    """
    print("=" * 50)
    print("开始处理文档并创建索引")
    print("=" * 50)
    
    file_path = "/workspaces/codespaces-blank/try_llamaindex/book/三国演义.txt"
    
    documents = load_documents(file_path)
    nodes = chunk_documents(documents, chunk_size=1024, chunk_overlap=20)
    
    # vector_index = create_vector_index(nodes)
    # summary_index = create_summary_index(nodes)
    # tree_index = create_tree_index(nodes)
    # keyword_index = create_keyword_index(nodes)
    property_graph_index = create_property_graph_index(nodes)
    
    print("\n" + "=" * 50)
    print("所有索引创建完成！")
    print("=" * 50)
    print("\n索引存储位置：")
    print("- 向量索引: Pinecone")
    print("- 摘要索引: MongoDB")
    print("- 树索引: MongoDB")
    print("- 关键词索引: MongoDB")
    print("- 属性图索引: Neo4j")
    
    return {
        "vector_index": vector_index,
        "summary_index": summary_index,
        "tree_index": tree_index,
        "keyword_index": keyword_index,
        "property_graph_index": property_graph_index
    }


if __name__ == "__main__":
    indexes = main()
