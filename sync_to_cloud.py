import os
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from storage import StorageManager
from models import llm, embed_model

# 本地存储目录
LOCAL_STORAGE_DIR = "./storage"


def check_local_index_exists(index_type):
    """
    检查本地索引是否存在
    
    Args:
        index_type (str): 索引类型（vector_index, summary_index, tree_index, keyword_index, property_graph_index）
    
    Returns:
        bool: 索引是否存在
    """
    local_dir = os.path.join(LOCAL_STORAGE_DIR, index_type)
    return os.path.exists(local_dir)


def load_vector_index_from_local():
    """
    从本地加载向量索引
    
    Returns:
        VectorStoreIndex: 向量索引实例，如果不存在则返回 None
    """
    print("=== 从本地加载向量索引 ===")
    
    if not check_local_index_exists("vector_index"):
        print("本地向量索引不存在")
        return None
    
    try:
        local_dir = os.path.join(LOCAL_STORAGE_DIR, "vector_index")
        storage_context = StorageContext.from_defaults(persist_dir=local_dir)
        index = load_index_from_storage(storage_context)
        print(f"向量索引已从本地加载：{local_dir}")
        return index
    except Exception as e:
        print(f"从本地加载向量索引失败：{e}")
        return None


def load_summary_index_from_local():
    """
    从本地加载摘要索引
    
    Returns:
        SummaryIndex: 摘要索引实例，如果不存在则返回 None
    """
    print("=== 从本地加载摘要索引 ===")
    
    if not check_local_index_exists("summary_index"):
        print("本地摘要索引不存在")
        return None
    
    try:
        local_dir = os.path.join(LOCAL_STORAGE_DIR, "summary_index")
        storage_context = StorageContext.from_defaults(persist_dir=local_dir)
        index = load_index_from_storage(storage_context)
        print(f"摘要索引已从本地加载：{local_dir}")
        return index
    except Exception as e:
        print(f"从本地加载摘要索引失败：{e}")
        return None


def load_tree_index_from_local():
    """
    从本地加载树索引
    
    Returns:
        TreeIndex: 树索引实例，如果不存在则返回 None
    """
    print("=== 从本地加载树索引 ===")
    
    if not check_local_index_exists("tree_index"):
        print("本地树索引不存在")
        return None
    
    try:
        local_dir = os.path.join(LOCAL_STORAGE_DIR, "tree_index")
        storage_context = StorageContext.from_defaults(persist_dir=local_dir)
        index = load_index_from_storage(storage_context)
        print(f"树索引已从本地加载：{local_dir}")
        return index
    except Exception as e:
        print(f"从本地加载树索引失败：{e}")
        return None


def load_keyword_index_from_local():
    """
    从本地加载关键词索引
    
    Returns:
        KeywordTableIndex: 关键词索引实例，如果不存在则返回 None
    """
    print("=== 从本地加载关键词索引 ===")
    
    if not check_local_index_exists("keyword_index"):
        print("本地关键词索引不存在")
        return None
    
    try:
        local_dir = os.path.join(LOCAL_STORAGE_DIR, "keyword_index")
        storage_context = StorageContext.from_defaults(persist_dir=local_dir)
        index = load_index_from_storage(storage_context)
        print(f"关键词索引已从本地加载：{local_dir}")
        return index
    except Exception as e:
        print(f"从本地加载关键词索引失败：{e}")
        return None


def load_property_graph_index_from_local():
    """
    从本地加载属性图索引
    
    Returns:
        PropertyGraphIndex: 属性图索引实例，如果不存在则返回 None
    """
    print("=== 从本地加载属性图索引 ===")
    
    if not check_local_index_exists("property_graph_index"):
        print("本地属性图索引不存在")
        return None
    
    try:
        local_dir = os.path.join(LOCAL_STORAGE_DIR, "property_graph_index")
        storage_context = StorageContext.from_defaults(persist_dir=local_dir)
        index = load_index_from_storage(storage_context)
        print(f"属性图索引已从本地加载：{local_dir}")
        return index
    except Exception as e:
        print(f"从本地加载属性图索引失败：{e}")
        return None


def sync_vector_index_to_cloud(index):
    """
    将向量索引同步到云端（Pinecone）
    
    Args:
        index (VectorStoreIndex): 向量索引实例
    
    Returns:
        bool: 同步是否成功
    """
    if index is None:
        print("没有向量索引需要同步")
        return False
    
    print("\n=== 同步向量索引到 Pinecone ===")
    
    try:
        storage_manager = StorageManager()
        storage_context = storage_manager.get_vector_storage_context()
        
        # 重新创建索引并保存到云端
        # 注意：这里需要从索引中提取文档或节点
        # 由于 LlamaIndex 的限制，我们使用索引的查询功能来重新创建
        print("开始同步向量索引...")
        
        # 获取所有节点
        nodes = list(index.docstore.docs.values())
        
        # 创建新的索引并保存到云端
        new_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print("向量索引已成功同步到 Pinecone")
        return True
        
    except Exception as e:
        print(f"同步向量索引到 Pinecone 失败：{e}")
        return False


def sync_summary_index_to_cloud(index):
    """
    将摘要索引同步到云端（MongoDB）
    
    Args:
        index (SummaryIndex): 摘要索引实例
    
    Returns:
        bool: 同步是否成功
    """
    if index is None:
        print("没有摘要索引需要同步")
        return False
    
    print("\n=== 同步摘要索引到 MongoDB ===")
    
    try:
        storage_manager = StorageManager()
        storage_context = storage_manager.get_summary_index_storage_context()
        
        # 获取所有节点
        nodes = list(index.docstore.docs.values())
        
        # 创建新的索引并保存到云端
        new_index = SummaryIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )
        
        print("摘要索引已成功同步到 MongoDB")
        return True
        
    except Exception as e:
        print(f"同步摘要索引到 MongoDB 失败：{e}")
        return False


def sync_tree_index_to_cloud(index):
    """
    将树索引同步到云端（MongoDB）
    
    Args:
        index (TreeIndex): 树索引实例
    
    Returns:
        bool: 同步是否成功
    """
    if index is None:
        print("没有树索引需要同步")
        return False
    
    print("\n=== 同步树索引到 MongoDB ===")
    
    try:
        storage_manager = StorageManager()
        storage_context = storage_manager.get_tree_index_storage_context()
        
        # 获取所有节点
        nodes = list(index.docstore.docs.values())
        
        # 创建新的索引并保存到云端
        new_index = TreeIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )
        
        print("树索引已成功同步到 MongoDB")
        return True
        
    except Exception as e:
        print(f"同步树索引到 MongoDB 失败：{e}")
        return False


def sync_keyword_index_to_cloud(index):
    """
    将关键词索引同步到云端（MongoDB）
    
    Args:
        index (KeywordTableIndex): 关键词索引实例
    
    Returns:
        bool: 同步是否成功
    """
    if index is None:
        print("没有关键词索引需要同步")
        return False
    
    print("\n=== 同步关键词索引到 MongoDB ===")
    
    try:
        storage_manager = StorageManager()
        storage_context = storage_manager.get_keyword_index_storage_context()
        
        # 获取所有节点
        nodes = list(index.docstore.docs.values())
        
        # 创建新的索引并保存到云端
        new_index = KeywordTableIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )
        
        print("关键词索引已成功同步到 MongoDB")
        return True
        
    except Exception as e:
        print(f"同步关键词索引到 MongoDB 失败：{e}")
        return False


def sync_property_graph_index_to_cloud(index):
    """
    将属性图索引同步到云端（Neo4j）
    
    Args:
        index (PropertyGraphIndex): 属性图索引实例
    
    Returns:
        bool: 同步是否成功
    """
    if index is None:
        print("没有属性图索引需要同步")
        return False
    
    print("\n=== 同步属性图索引到 Neo4j ===")
    
    try:
        storage_manager = StorageManager()
        graph_store = storage_manager.get_neo4j_property_graph_store()
        storage_context = storage_manager.get_neo4j_storage_context(embed_dim=768)
        
        # 获取所有节点
        nodes = list(index.docstore.docs.values())
        
        # 创建新的索引并保存到云端
        kg_extractors = [
            SchemaLLMPathExtractor(llm=llm)
        ]
        
        new_index = PropertyGraphIndex(
            nodes=nodes,
            storage_context=storage_context,
            llm=llm,
            embed_model=embed_model,
            kg_extractors=kg_extractors,
            property_graph_store=graph_store,
            show_progress=True
        )
        
        print("属性图索引已成功同步到 Neo4j")
        return True
        
    except Exception as e:
        print(f"同步属性图索引到 Neo4j 失败：{e}")
        return False


def main():
    """
    主函数：从本地加载索引并同步到云端
    """
    print("=" * 50)
    print("开始同步本地索引到云端")
    print("=" * 50)
    print(f"\n本地存储目录: {LOCAL_STORAGE_DIR}")
    
    # 检查本地存储目录是否存在
    if not os.path.exists(LOCAL_STORAGE_DIR):
        print(f"本地存储目录不存在：{LOCAL_STORAGE_DIR}")
        return
    
    # 加载本地索引
    vector_index = load_vector_index_from_local()
    summary_index = load_summary_index_from_local()
    tree_index = load_tree_index_from_local()
    keyword_index = load_keyword_index_from_local()
    property_graph_index = load_property_graph_index_from_local()
    
    # 同步到云端
    results = []
    
    if vector_index is not None:
        result = sync_vector_index_to_cloud(vector_index)
        results.append(("向量索引", result))
    
    if summary_index is not None:
        result = sync_summary_index_to_cloud(summary_index)
        results.append(("摘要索引", result))
    
    if tree_index is not None:
        result = sync_tree_index_to_cloud(tree_index)
        results.append(("树索引", result))
    
    if keyword_index is not None:
        result = sync_keyword_index_to_cloud(keyword_index)
        results.append(("关键词索引", result))
    
    if property_graph_index is not None:
        result = sync_property_graph_index_to_cloud(property_graph_index)
        results.append(("属性图索引", result))
    
    # 显示结果
    print("\n" + "=" * 50)
    print("同步完成！")
    print("=" * 50)
    print("\n同步结果：")
    for index_name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"- {index_name}: {status}")
    
    # 统计
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n总计: {success_count}/{total_count} 个索引同步成功")


if __name__ == "__main__":
    main()
