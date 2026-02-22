import os
import yaml
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class StorageManager:
    """
    存储管理器类，用于统一管理不同存储后端的连接和配置
    
    支持的存储后端：
    - Pinecone：用于存储向量索引
    - MongoDB：用于存储文档和索引元数据（摘要索引、树索引、关键词索引）
    - Neo4j：用于存储属性图索引
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化存储管理器
        
        Args:
            config_path (str): 配置文件路径，默认为 config.yaml
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.storage_config = self.config["storage"]
    
    def get_pinecone_storage_context(self, index_name=None):
        """
        获取 Pinecone 存储上下文
        
        用于向量索引的存储
        
        Args:
            index_name (str, optional): 索引名称，如果不指定则使用配置文件中的默认值
        
        Returns:
            StorageContext: 包含 Pinecone 向量存储的存储上下文
        """
        pinecone_config = self.storage_config["pinecone"]
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pc = Pinecone(api_key=api_key)
        actual_index_name = index_name or pinecone_config["index_name"]
        
        try:
            pc.describe_index(actual_index_name)
        except Exception:
            pc.create_index(
                name=actual_index_name,
                dimension=pinecone_config["dimension"],
                metric=pinecone_config["metric"],
                spec=ServerlessSpec(
                    cloud=pinecone_config["cloud"],
                    region=pinecone_config["region"]
                )
            )
        
        pinecone_index = pc.Index(actual_index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context
    
    def get_mongodb_storage_context(self, namespace="default"):
        """
        获取 MongoDB 存储上下文
        
        用于文档存储和索引元数据存储
        
        Args:
            namespace (str): 命名空间，用于区分不同的索引，默认为 "default"
        
        Returns:
            StorageContext: 包含 MongoDB 文档存储和索引存储的存储上下文
        """
        mongo_config = self.storage_config["mongodb"]
        mongo_uri = os.environ.get("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables")
        
        docstore = MongoDocumentStore.from_uri(
            uri=mongo_uri,
            db_name=mongo_config["db_name"],
            namespace=namespace
        )
        
        index_store = MongoIndexStore.from_uri(
            uri=mongo_uri,
            db_name=mongo_config["db_name"],
            namespace=namespace
        )
        
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store
        )
        return storage_context
    
    def get_neo4j_storage_context(self, embed_dim=768):
        """
        获取 Neo4j 存储上下文
        
        用于属性图索引的存储
        
        Args:
            embed_dim (int): 嵌入向量的维度，默认为 768（与 Google GenAI Embedding 一致）
        
        Returns:
            StorageContext: 包含 Neo4j 向量存储的存储上下文
        """
        neo4j_config = self.storage_config["neo4j"]
        url = os.environ.get("NEO4J_URI")
        username = os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")
        database = neo4j_config["database"]
        
        if not url or not username or not password:
            raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in environment variables")
        
        from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
        vector_store = Neo4jVectorStore(
            username=username,
            password=password,
            url=url,
            database=database,
            embedding_dimension=embed_dim,
            index_name=neo4j_config["index_name"],
            text_node_property=neo4j_config["text_node_property"],
            hybrid_search=neo4j_config["hybrid_search"]
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context
    
    def get_neo4j_property_graph_store(self):
        """
        获取 Neo4j 属性图存储实例
        
        用于存储和管理知识图谱（属性图索引）
        
        Returns:
            Neo4jPropertyGraphStore: Neo4j 属性图存储实例
        """
        neo4j_config = self.storage_config["neo4j"]
        url = os.environ.get("NEO4J_URI")
        username = os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")
        database = neo4j_config["database"]
        
        if not url or not username or not password:
            raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in environment variables")
        
        graph_store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url,
            database=database
        )
        return graph_store
