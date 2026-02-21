import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core import load_index_from_storage
from storage import StorageManager
from models import llm, embed_model

nest_asyncio.apply()

# 本地存储目录
LOCAL_STORAGE_DIR = "./storage"


def ensure_local_storage_dirs():
    """
    确保本地存储目录存在
    
    创建所有需要的本地存储目录
    """
    dirs = [
        os.path.join(LOCAL_STORAGE_DIR, "vector_index"),
        os.path.join(LOCAL_STORAGE_DIR, "summary_index"),
        os.path.join(LOCAL_STORAGE_DIR, "tree_index"),
        os.path.join(LOCAL_STORAGE_DIR, "keyword_index"),
        os.path.join(LOCAL_STORAGE_DIR, "property_graph_index")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"本地存储目录已准备：{LOCAL_STORAGE_DIR}")


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


def generate_vector_index(documents):
    """
    生成向量索引
    
    Args:
        documents (list): 文档列表
    
    Returns:
        VectorStoreIndex: 向量索引实例
    """
    local_storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=local_storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    return index


def save_vector_index_to_local(index):
    """
    将向量索引保存到本地
    
    Args:
        index (VectorStoreIndex): 向量索引实例
    
    Returns:
        str: 本地存储目录路径
    """
    local_dir = os.path.join(LOCAL_STORAGE_DIR, "vector_index")
    index.storage_context.persist(persist_dir=local_dir)
    print(f"向量索引已保存到本地：{local_dir}")
    return local_dir


def sync_vector_index_to_cloud(index):
    """
    将向量索引同步到云端（Pinecone）
    
    Args:
        index (VectorStoreIndex): 向量索引实例
    
    Returns:
        VectorStoreIndex: 云端索引实例，如果失败则返回原索引
    """
    try:
        storage_manager = StorageManager()
        cloud_storage_context = storage_manager.get_vector_storage_context()
        
        nodes = list(index.docstore.docs.values())
        cloud_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=cloud_storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        print("向量索引已成功同步到 Pinecone")
        return cloud_index
        
    except Exception as e:
        print(f"同步到 Pinecone 失败：{e}")
        print("向量索引仅保存在本地")
        return index


def create_vector_index(documents):
    """
    创建向量索引：生成 → 保存到本地 → 同步到云端
    
    Args:
        documents (list): 文档列表
    
    Returns:
        VectorStoreIndex: 向量索引实例
    """
    print("\n=== 创建向量索引 ===")
    
    index = generate_vector_index(documents)
    save_vector_index_to_local(index)
    index = sync_vector_index_to_cloud(index)
    
    return index


def generate_summary_index(documents):
    """
    生成摘要索引
    
    Args:
        documents (list): 文档列表
    
    Returns:
        SummaryIndex: 摘要索引实例
    """
    local_storage_context = StorageContext.from_defaults()
    index = SummaryIndex.from_documents(
        documents,
        storage_context=local_storage_context,
        show_progress=True
    )
    return index


def save_summary_index_to_local(index):
    """
    将摘要索引保存到本地
    
    Args:
        index (SummaryIndex): 摘要索引实例
    
    Returns:
        str: 本地存储目录路径
    """
    local_dir = os.path.join(LOCAL_STORAGE_DIR, "summary_index")
    index.storage_context.persist(persist_dir=local_dir)
    print(f"摘要索引已保存到本地：{local_dir}")
    return local_dir


def sync_summary_index_to_cloud(index):
    """
    将摘要索引同步到云端（MongoDB）
    
    Args:
        index (SummaryIndex): 摘要索引实例
    
    Returns:
        SummaryIndex: 云端索引实例，如果失败则返回原索引
    """
    try:
        storage_manager = StorageManager()
        cloud_storage_context = storage_manager.get_summary_index_storage_context()
        
        nodes = list(index.docstore.docs.values())
        cloud_index = SummaryIndex(
            nodes=nodes,
            storage_context=cloud_storage_context,
            show_progress=True
        )
        print("摘要索引已成功同步到 MongoDB")
        return cloud_index
        
    except Exception as e:
        print(f"同步到 MongoDB 失败：{e}")
        print("摘要索引仅保存在本地")
        return index


def create_summary_index(documents):
    """
    创建摘要索引：生成 → 保存到本地 → 同步到云端
    
    Args:
        documents (list): 文档列表
    
    Returns:
        SummaryIndex: 摘要索引实例
    """
    print("\n=== 创建摘要索引 ===")
    
    index = generate_summary_index(documents)
    save_summary_index_to_local(index)
    index = sync_summary_index_to_cloud(index)
    
    return index


def generate_tree_index(documents):
    """
    生成树索引
    
    Args:
        documents (list): 文档列表
    
    Returns:
        TreeIndex: 树索引实例
    """
    local_storage_context = StorageContext.from_defaults()
    index = TreeIndex.from_documents(
        documents,
        storage_context=local_storage_context,
        show_progress=True
    )
    return index


def save_tree_index_to_local(index):
    """
    将树索引保存到本地
    
    Args:
        index (TreeIndex): 树索引实例
    
    Returns:
        str: 本地存储目录路径
    """
    local_dir = os.path.join(LOCAL_STORAGE_DIR, "tree_index")
    index.storage_context.persist(persist_dir=local_dir)
    print(f"树索引已保存到本地：{local_dir}")
    return local_dir


def sync_tree_index_to_cloud(index):
    """
    将树索引同步到云端（MongoDB）
    
    Args:
        index (TreeIndex): 树索引实例
    
    Returns:
        TreeIndex: 云端索引实例，如果失败则返回原索引
    """
    try:
        storage_manager = StorageManager()
        cloud_storage_context = storage_manager.get_tree_index_storage_context()
        
        nodes = list(index.docstore.docs.values())
        cloud_index = TreeIndex(
            nodes=nodes,
            storage_context=cloud_storage_context,
            show_progress=True
        )
        print("树索引已成功同步到 MongoDB")
        return cloud_index
        
    except Exception as e:
        print(f"同步到 MongoDB 失败：{e}")
        print("树索引仅保存在本地")
        return index


def create_tree_index(documents):
    """
    创建树索引：生成 → 保存到本地 → 同步到云端
    
    Args:
        documents (list): 文档列表
    
    Returns:
        TreeIndex: 树索引实例
    """
    print("\n=== 创建树索引 ===")
    
    index = generate_tree_index(documents)
    save_tree_index_to_local(index)
    index = sync_tree_index_to_cloud(index)
    
    return index


def generate_keyword_index(documents):
    """
    生成关键词索引
    
    Args:
        documents (list): 文档列表
    
    Returns:
        KeywordTableIndex: 关键词索引实例
    """
    local_storage_context = StorageContext.from_defaults()
    index = KeywordTableIndex.from_documents(
        documents,
        storage_context=local_storage_context,
        show_progress=True
    )
    return index


def save_keyword_index_to_local(index):
    """
    将关键词索引保存到本地
    
    Args:
        index (KeywordTableIndex): 关键词索引实例
    
    Returns:
        str: 本地存储目录路径
    """
    local_dir = os.path.join(LOCAL_STORAGE_DIR, "keyword_index")
    index.storage_context.persist(persist_dir=local_dir)
    print(f"关键词索引已保存到本地：{local_dir}")
    return local_dir


def sync_keyword_index_to_cloud(index):
    """
    将关键词索引同步到云端（MongoDB）
    
    Args:
        index (KeywordTableIndex): 关键词索引实例
    
    Returns:
        KeywordTableIndex: 云端索引实例，如果失败则返回原索引
    """
    try:
        storage_manager = StorageManager()
        cloud_storage_context = storage_manager.get_keyword_index_storage_context()
        
        nodes = list(index.docstore.docs.values())
        cloud_index = KeywordTableIndex(
            nodes=nodes,
            storage_context=cloud_storage_context,
            show_progress=True
        )
        print("关键词索引已成功同步到 MongoDB")
        return cloud_index
        
    except Exception as e:
        print(f"同步到 MongoDB 失败：{e}")
        print("关键词索引仅保存在本地")
        return index


def create_keyword_index(documents):
    """
    创建关键词索引：生成 → 保存到本地 → 同步到云端
    
    Args:
        documents (list): 文档列表
    
    Returns:
        KeywordTableIndex: 关键词索引实例
    """
    print("\n=== 创建关键词索引 ===")
    
    index = generate_keyword_index(documents)
    save_keyword_index_to_local(index)
    index = sync_keyword_index_to_cloud(index)
    
    return index


def generate_property_graph_index(documents):
    """
    生成属性图索引
    
    Args:
        documents (list): 文档列表
    
    Returns:
        PropertyGraphIndex: 属性图索引实例
    """
    kg_extractors = [
        SchemaLLMPathExtractor(
            llm=llm
        )
    ]
    
    local_storage_context = StorageContext.from_defaults()
    index = PropertyGraphIndex.from_documents(
        documents,
        storage_context=local_storage_context,
        llm=llm,
        embed_model=embed_model,
        kg_extractors=kg_extractors,
        show_progress=True
    )
    return index


def save_property_graph_index_to_local(index):
    """
    将属性图索引保存到本地
    
    Args:
        index (PropertyGraphIndex): 属性图索引实例
    
    Returns:
        str: 本地存储目录路径
    """
    local_dir = os.path.join(LOCAL_STORAGE_DIR, "property_graph_index")
    index.storage_context.persist(persist_dir=local_dir)
    print(f"属性图索引已保存到本地：{local_dir}")
    return local_dir


def sync_property_graph_index_to_cloud(index):
    """
    将属性图索引同步到云端（Neo4j）
    
    Args:
        index (PropertyGraphIndex): 属性图索引实例
    
    Returns:
        PropertyGraphIndex: 云端索引实例，如果失败则返回原索引
    """
    try:
        storage_manager = StorageManager()
        graph_store = storage_manager.get_neo4j_property_graph_store()
        cloud_storage_context = storage_manager.get_neo4j_storage_context(embed_dim=768)
        
        kg_extractors = [
            SchemaLLMPathExtractor(llm=llm)
        ]
        
        nodes = list(index.docstore.docs.values())
        cloud_index = PropertyGraphIndex(
            nodes=nodes,
            storage_context=cloud_storage_context,
            llm=llm,
            embed_model=embed_model,
            kg_extractors=kg_extractors,
            property_graph_store=graph_store,
            show_progress=True
        )
        print("属性图索引已成功同步到 Neo4j")
        return cloud_index
        
    except Exception as e:
        print(f"同步到 Neo4j 失败：{e}")
        print("属性图索引仅保存在本地")
        return index


def create_property_graph_index(documents):
    """
    创建属性图索引：生成 → 保存到本地 → 同步到云端
    
    Args:
        documents (list): 文档列表
    
    Returns:
        PropertyGraphIndex: 属性图索引实例
    """
    print("\n=== 创建属性图索引 ===")
    
    index = generate_property_graph_index(documents)
    save_property_graph_index_to_local(index)
    index = sync_property_graph_index_to_cloud(index)
    
    return index


def main():
    """
    主函数：加载文档、分块、创建各种索引并存储
    """
    print("=" * 50)
    print("开始处理文档并创建索引")
    print("=" * 50)
    
    # 确保本地存储目录存在
    ensure_local_storage_dirs()
    
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
    print("- 向量索引: 本地 + Pinecone")
    print("- 摘要索引: 本地 + MongoDB")
    print("- 树索引: 本地 + MongoDB")
    print("- 关键词索引: 本地 + MongoDB")
    print("- 属性图索引: 本地 + Neo4j")
    print(f"\n本地存储目录: {LOCAL_STORAGE_DIR}")
    
    return {
        "vector_index": vector_index,
        "summary_index": summary_index,
        "tree_index": tree_index,
        "keyword_index": keyword_index,
        "property_graph_index": property_graph_index
    }


if __name__ == "__main__":
    indexes = main()
