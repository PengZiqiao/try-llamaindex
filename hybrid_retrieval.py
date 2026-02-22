import os
import nest_asyncio
from llama_index.core import (
    VectorStoreIndex,
    DocumentSummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    StorageContext,
    Settings
)
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.tools import QueryEngineTool
from llama_index.core import PromptTemplate
from storage import StorageManager
from models import llm, embed_model

nest_asyncio.apply()

# 设置全局配置
Settings.llm = llm
Settings.embed_model = embed_model


def load_vector_index():
    """
    从 Pinecone 加载向量索引
    
    Returns:
        VectorStoreIndex: 向量索引实例
    """
    print("=== 加载向量索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_pinecone_storage_context()
    
    # 从存储上下文加载索引
    index = VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        embed_model=embed_model
    )
    print("向量索引加载成功")
    return index


def load_summary_index():
    """
    从 MongoDB 加载摘要索引
    
    Returns:
        DocumentSummaryIndex: 摘要索引实例
    """
    print("=== 加载摘要索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_mongodb_storage_context(namespace="summary_index")
    
    # 从存储上下文加载索引
    index = DocumentSummaryIndex.from_documents([], storage_context=storage_context)
    print("摘要索引加载成功")
    return index


def load_tree_index():
    """
    从 MongoDB 加载树索引
    
    Returns:
        TreeIndex: 树索引实例
    """
    print("=== 加载树索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_mongodb_storage_context(namespace="tree_index")
    
    # 从存储上下文加载索引
    index = TreeIndex.from_documents([], storage_context=storage_context)
    print("树索引加载成功")
    return index


def load_keyword_index():
    """
    从 MongoDB 加载关键词索引
    
    Returns:
        KeywordTableIndex: 关键词索引实例
    """
    print("=== 加载关键词索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_mongodb_storage_context(namespace="keyword_index")
    
    # 从存储上下文加载索引
    index = KeywordTableIndex.from_documents([], storage_context=storage_context)
    print("关键词索引加载成功")
    return index


def load_property_graph_index():
    """
    从 Neo4j 加载属性图索引
    
    Returns:
        PropertyGraphIndex: 属性图索引实例
    """
    print("=== 加载属性图索引 ===")
    storage_manager = StorageManager()
    storage_context = storage_manager.get_neo4j_storage_context(embed_dim=768)
    graph_store = storage_manager.get_neo4j_property_graph_store()
    
    # 从存储上下文加载索引
    index = PropertyGraphIndex.from_documents(
        [],
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store
    )
    print("属性图索引加载成功")
    return index


def create_hybrid_query_engine():
    """
    创建混合查询引擎，结合多种索引进行检索
    
    使用 Router Query Engine，根据查询类型自动选择最合适的索引
    
    Returns:
        RouterQueryEngine: 混合查询引擎实例
    """
    print("\n=== 创建混合查询引擎 ===")
    
    # 加载所有索引
    vector_index = load_vector_index()
    summary_index = load_summary_index()
    tree_index = load_tree_index()
    keyword_index = load_keyword_index()
    property_graph_index = load_property_graph_index()
    
    # 定义查询提示模板
    QA_PROMPT_TMPL = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given context information and not prior knowledge, "
        "answer the question. If the answer is not in the context, inform "
        "the user that you can't answer the question - DO NOT MAKE UP AN ANSWER.\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)
    
    # 为每个索引创建查询引擎
    vector_query_engine = vector_index.as_query_engine(
        text_qa_template=QA_PROMPT,
        similarity_top_k=5
    )
    
    summary_query_engine = summary_index.as_query_engine(
        text_qa_template=QA_PROMPT
    )
    
    tree_query_engine = tree_index.as_query_engine(
        text_qa_template=QA_PROMPT
    )
    
    keyword_query_engine = keyword_index.as_query_engine(
        text_qa_template=QA_PROMPT
    )
    
    property_graph_query_engine = property_graph_index.as_query_engine(
        text_qa_template=QA_PROMPT,
        include_text=True
    )
    
    # 创建查询引擎工具
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="向量索引：适用于语义相似性搜索，适合查找与问题语义相关的内容"
    )
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="摘要索引：适用于获取文档的整体摘要和概述"
    )
    
    tree_tool = QueryEngineTool.from_defaults(
        query_engine=tree_query_engine,
        description="树索引：适用于层次化查询，可以快速定位到相关章节"
    )
    
    keyword_tool = QueryEngineTool.from_defaults(
        query_engine=keyword_query_engine,
        description="关键词索引：适用于基于关键词的精确匹配搜索"
    )
    
    property_graph_tool = QueryEngineTool.from_defaults(
        query_engine=property_graph_query_engine,
        description="属性图索引：适用于实体关系查询和知识图谱检索"
    )
    
    # 定义综合提示模板
    TREE_SUMMARIZE_PROMPT_TMPL = (
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given information from multiple sources and not prior knowledge, "
        "answer the question. If the answer is not in the context, inform "
        "the user that you can't answer the question.\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    
    # 创建综合器
    tree_summarize = TreeSummarize(
        summary_template=PromptTemplate(TREE_SUMMARIZE_PROMPT_TMPL),
        llm=llm
    )
    
    # 创建路由查询引擎
    # 使用 LLMMultiSelector 可以让 LLM 选择多个最合适的查询引擎
    query_engine = RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(),
        query_engine_tools=[
            vector_tool,
            summary_tool,
            tree_tool,
            keyword_tool,
            property_graph_tool
        ],
        summarizer=tree_summarize
    )
    
    print("混合查询引擎创建成功！")
    print("\n可用的检索策略：")
    print("- 向量索引：语义相似性搜索")
    print("- 摘要索引：文档摘要和概述")
    print("- 树索引：层次化查询")
    print("- 关键词索引：关键词精确匹配")
    print("- 属性图索引：实体关系查询")
    
    return query_engine


def query_hybrid(query_engine, question):
    """
    使用混合查询引擎进行查询
    
    Args:
        query_engine: 混合查询引擎实例
        question: 用户问题
    
    Returns:
        查询结果
    """
    print(f"\n=== 查询问题：{question} ===")
    response = query_engine.query(question)
    return response


def main():
    """
    主函数：加载索引、创建混合查询引擎、进行查询
    """
    print("=" * 60)
    print("混合检索系统")
    print("=" * 60)
    
    # 创建混合查询引擎
    query_engine = create_hybrid_query_engine()
    
    # 示例查询
    questions = [
        "三国演义的主要人物有哪些？",
        "诸葛亮的主要事迹是什么？",
        "赤壁之战的经过是怎样的？",
        "刘备和关羽的关系如何？"
    ]
    
    print("\n" + "=" * 60)
    print("开始查询")
    print("=" * 60)
    
    for question in questions:
        response = query_hybrid(query_engine, question)
        print(f"\n回答：{response}")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("查询完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
