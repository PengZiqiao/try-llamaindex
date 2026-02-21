from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.graph_stores import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex
from storage import StorageManager
from models import llm, embed_model


def create_vector_index(documents):
    storage_manager = StorageManager()
    storage_context = storage_manager.get_vector_storage_context()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    return index


def create_summary_index(documents):
    storage_manager = StorageManager()
    storage_context = storage_manager.get_summary_index_storage_context()
    index = SummaryIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    return index


def create_tree_index(documents):
    storage_manager = StorageManager()
    storage_context = storage_manager.get_tree_index_storage_context()
    index = TreeIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    return index


def create_keyword_index(documents):
    storage_manager = StorageManager()
    storage_context = storage_manager.get_keyword_index_storage_context()
    index = KeywordTableIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    return index


def create_property_graph_index(documents):
    storage_manager = StorageManager()
    graph_store = storage_manager.get_neo4j_property_graph_store()
    index = PropertyGraphIndex.from_documents(
        documents,
        storage_context=storage_manager.get_neo4j_storage_context(embed_dim=3072),
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store,
        show_progress=True
    )
    return index


if __name__ == "__main__":
    documents = SimpleDirectoryReader("./book").load_data()
    
    vector_index = create_vector_index(documents)
    print("Vector index created with Pinecone")
    
    summary_index = create_summary_index(documents)
    print("Summary index created with MongoDB")
    
    tree_index = create_tree_index(documents)
    print("Tree index created with MongoDB")
    
    keyword_index = create_keyword_index(documents)
    print("Keyword index created with MongoDB")
    
    property_graph_index = create_property_graph_index(documents)
    print("Property graph index created with Neo4j")
