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
    
    def _get_pinecone_vector_store(self, index_name=None):
        """
        获取 Pinecone 向量存储实例（内部方法）
        
        如果指定的索引不存在，会自动创建新索引
        
        Args:
            index_name (str, optional): 索引名称，如果不指定则使用配置文件中的默认值
        
        Returns:
            PineconeVectorStore: Pinecone 向量存储实例
        
        Raises:
            ValueError: 当环境变量中未设置 PINECONE_API_KEY 时抛出
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
        return vector_store
    
    def _get_mongodb_docstore(self, index_type="summary"):
        """
        获取 MongoDB 文档存储实例（内部方法）
        
        用于存储文档节点（Node 对象），支持三种索引类型
        
        Args:
            index_type (str): 索引类型，可选值为：
                - "summary": 摘要索引
                - "tree": 树索引
                - "keyword": 关键词索引
                默认为 "summary"
        
        Returns:
            MongoDocumentStore: MongoDB 文档存储实例
        
        Raises:
            ValueError: 当环境变量中未设置 MONGO_URI 或 index_type 无效时抛出
        """
        mongo_config = self.storage_config["mongodb"]
        mongo_uri = os.environ.get("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables")
        
        if index_type == "summary":
            collection_config = mongo_config["summary_index"]
        elif index_type == "tree":
            collection_config = mongo_config["tree_index"]
        elif index_type == "keyword":
            collection_config = mongo_config["keyword_index"]
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        docstore = MongoDocumentStore.from_uri(
            uri=mongo_uri,
            db_name=mongo_config["db_name"],
            namespace=collection_config["namespace"]
        )
        return docstore
    
    def _get_mongodb_index_store(self, index_type="summary"):
        """
        获取 MongoDB 索引存储实例（内部方法）
        
        用于存储索引元数据（轻量级的索引状态信息），支持三种索引类型
        
        Args:
            index_type (str): 索引类型，可选值为：
                - "summary": 摘要索引
                - "tree": 树索引
                - "keyword": 关键词索引
                默认为 "summary"
        
        Returns:
            MongoIndexStore: MongoDB 索引存储实例
        
        Raises:
            ValueError: 当环境变量中未设置 MONGO_URI 或 index_type 无效时抛出
        """
        mongo_config = self.storage_config["mongodb"]
        mongo_uri = os.environ.get("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables")
        
        if index_type == "summary":
            collection_config = mongo_config["summary_index"]
        elif index_type == "tree":
            collection_config = mongo_config["tree_index"]
        elif index_type == "keyword":
            collection_config = mongo_config["keyword_index"]
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        index_store = MongoIndexStore.from_uri(
            uri=mongo_uri,
            db_name=mongo_config["db_name"],
            namespace=collection_config["namespace"]
        )
        return index_store
    
    def get_neo4j_property_graph_store(self):
        """
        获取 Neo4j 属性图存储实例
        
        用于存储和管理知识图谱（属性图索引）
        
        Returns:
            Neo4jPropertyGraphStore: Neo4j 属性图存储实例
        """
        neo4j_config = self.storage_config["neo4j"]
        url = os.environ.get("NEO4J_URI", neo4j_config["url"])
        username = os.environ.get("NEO4J_USERNAME", neo4j_config["username"])
        password = os.environ.get("NEO4J_PASSWORD", neo4j_config["password"])
        
        graph_store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url
        )
        return graph_store
    
    def get_vector_storage_context(self, index_name=None):
        """
        获取向量索引的存储上下文
        
        使用 Pinecone 作为向量存储后端
        
        Args:
            index_name (str, optional): 索引名称，如果不指定则使用配置文件中的默认值
        
        Returns:
            StorageContext: 包含 Pinecone 向量存储的存储上下文
        """
        vector_store = self._get_pinecone_vector_store(index_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context
    
    def get_summary_index_storage_context(self):
        """
        获取摘要索引的存储上下文
        
        使用 MongoDB 作为文档存储和索引存储后端
        
        Returns:
            StorageContext: 包含 MongoDB 文档存储和索引存储的存储上下文
        """
        docstore = self._get_mongodb_docstore("summary")
        index_store = self._get_mongodb_index_store("summary")
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store
        )
        return storage_context
    
    def get_tree_index_storage_context(self):
        """
        获取树索引的存储上下文
        
        使用 MongoDB 作为文档存储和索引存储后端
        
        Returns:
            StorageContext: 包含 MongoDB 文档存储和索引存储的存储上下文
        """
        docstore = self._get_mongodb_docstore("tree")
        index_store = self._get_mongodb_index_store("tree")
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store
        )
        return storage_context
    
    def get_keyword_index_storage_context(self):
        """
        获取关键词索引的存储上下文
        
        使用 MongoDB 作为文档存储和索引存储后端
        
        Returns:
            StorageContext: 包含 MongoDB 文档存储和索引存储的存储上下文
        """
        docstore = self._get_mongodb_docstore("keyword")
        index_store = self._get_mongodb_index_store("keyword")
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store
        )
        return storage_context
    
    def get_neo4j_storage_context(self, embed_dim=768):
        """
        获取 Neo4j 存储上下文
        
        使用 Neo4j 作为向量存储后端，适用于属性图索引
        
        Args:
            embed_dim (int): 嵌入向量的维度，默认为 768（与 Google GenAI Embedding 一致）
        
        Returns:
            StorageContext: 包含 Neo4j 向量存储的存储上下文
        """
        neo4j_config = self.storage_config["neo4j"]
        url = os.environ.get("NEO4J_URI", neo4j_config["url"])
        username = os.environ.get("NEO4J_USERNAME", neo4j_config["username"])
        password = os.environ.get("NEO4J_PASSWORD", neo4j_config["password"])
        
        from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
        vector_store = Neo4jVectorStore(
            username=username,
            password=password,
            url=url,
            embed_dim=embed_dim,
            index_name=neo4j_config["index_name"],
            text_node_property=neo4j_config["text_node_property"],
            hybrid_search=neo4j_config["hybrid_search"]
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context
