import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from storage import StorageManager
from models import llm, embed_model

nest_asyncio.apply()


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
    return nodes


def create_vector_index(documents):
    """
    创建向量索引并存储到 Pinecone
    
    Args:
        documents (list): 文档列表
    
    Returns:
        VectorStoreIndex: 向量索引实例
    """
    print("\n=== 创建向量索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_vector_storage_context()
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print("向量索引创建成功，已存储到 Pinecone")
    return index


def create_summary_index(documents):
    """
    创建摘要索引并存储到 MongoDB
    
    Args:
        documents (list): 文档列表
    
    Returns:
        SummaryIndex: 摘要索引实例
    """
    print("\n=== 创建摘要索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_summary_index_storage_context()
    
    index = SummaryIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("摘要索引创建成功，已存储到 MongoDB")
    return index


def create_tree_index(documents):
    """
    创建树索引并存储到 MongoDB
    
    Args:
        documents (list): 文档列表
    
    Returns:
        TreeIndex: 树索引实例
    """
    print("\n=== 创建树索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_tree_index_storage_context()
    
    index = TreeIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("树索引创建成功，已存储到 MongoDB")
    return index


def create_keyword_index(documents):
    """
    创建关键词索引并存储到 MongoDB
    
    Args:
        documents (list): 文档列表
    
    Returns:
        KeywordTableIndex: 关键词索引实例
    """
    print("\n=== 创建关键词索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_keyword_index_storage_context()
    
    index = KeywordTableIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("关键词索引创建成功，已存储到 MongoDB")
    return index


def create_property_graph_index(documents):
    """
    创建属性图索引并存储到 Neo4j
    
    Args:
        documents (list): 文档列表
    
    Returns:
        PropertyGraphIndex: 属性图索引实例
    """
    print("\n=== 创建属性图索引 ===")
    storage_manager = StorageManager()
    graph_store = storage_manager.get_neo4j_property_graph_store()
    
    kg_extractors = [
        SchemaLLMPathExtractor(
            llm=llm
        )
    ]
    
    index = PropertyGraphIndex.from_documents(
        documents,
        storage_context=storage_manager.get_neo4j_storage_context(embed_dim=768),
        llm=llm,
        embed_model=embed_model,
        kg_extractors=kg_extractors,
        property_graph_store=graph_store,
        show_progress=True
    )
    print("属性图索引创建成功，已存储到 Neo4j")
    return index


def main():
    """
    主函数：加载文档、分块、创建各种索引并存储
    """
    print("=" * 50)
    print("开始处理文档并创建索引")
    print("=" * 50)
    
    file_path = "/workspaces/codespaces-blank/try_llamaindex/book/三国演义.txt"
    
    documents = load_documents(file_path)
    
    nodes = chunk_documents(documents, chunk_size=1024, chunk_overlap=20)
    
    vector_index = create_vector_index(documents)
    
    summary_index = create_summary_index(documents)
    
    tree_index = create_tree_index(documents)
    
    keyword_index = create_keyword_index(documents)
    
    property_graph_index = create_property_graph_index(documents)
    
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
