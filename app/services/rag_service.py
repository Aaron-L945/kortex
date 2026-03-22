import os
from typing import List, Optional

import os
from typing import List, Optional

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from config import settings
from models.schemas import DocumentMetadata, UserInfo
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters, FilterOperator

class RAGService:
    def __init__(self):
        self.embed_model = HuggingFaceEmbedding(model_name=settings.EMBED_MODEL)
        self.faiss_index = None
        self.vector_store = None
        self.index = None
        self._init_faiss_index()

    def _init_faiss_index(self):
        if not os.path.exists(settings.FAISS_INDEX_PATH) or not os.listdir(settings.FAISS_INDEX_PATH):
            os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
            # Initialize an empty FAISS index if it doesn't exist
            # Assuming embedding dimension for bge-large-zh-v1.5 is 1024
            d = 1024  
            self.faiss_index = faiss.IndexFlatL2(d)
            self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_documents(
                [], storage_context=self.storage_context, embed_model=self.embed_model
            )
            self.index.storage_context.persist(persist_dir=settings.FAISS_INDEX_PATH)
        else:
            # Load existing FAISS index
            self.storage_context = StorageContext.from_defaults(
                vector_store=FaissVectorStore.from_persist_dir(settings.FAISS_INDEX_PATH)
            )
            self.index = VectorStoreIndex.load_from_storage(
                storage_context=self.storage_context, embed_model=self.embed_model
            )

    def ingest_documents(self, documents: List[DocumentMetadata]):
        # Create TextNode from DocumentMetadata
        nodes = []
        for doc_meta in documents:
            node = TextNode(
                text=doc_meta.content,
                metadata={
                    "doc_id": doc_meta.doc_id,
                    "chunk_id": doc_meta.chunk_id,
                    "title": doc_meta.title,
                    "source": doc_meta.source,
                    "user_groups": doc_meta.user_groups,
                    "departments": doc_meta.departments,
                    "permission_level": doc_meta.permission_level,
                }
            )
            nodes.append(node)
        
        self.index.insert_nodes(nodes)
        self.index.storage_context.persist(persist_dir=settings.FAISS_INDEX_PATH)

    def retrieve(self, query: str, user: UserInfo, top_k: int = 5) -> List[DocumentMetadata]:
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            vector_store_query_mode="default",
            filters=self._build_permission_filters(user)
        )
        response = query_engine.query(query)
        
        retrieved_docs = []
        for node in response.source_nodes:
            metadata = node.metadata
            retrieved_docs.append(
                DocumentMetadata(
                    doc_id=metadata["doc_id"],
                    chunk_id=metadata["chunk_id"],
                    title=metadata["title"],
                    source=metadata["source"],
                    content=node.text,
                    user_groups=metadata["user_groups"],
                    departments=metadata["departments"],
                    permission_level=metadata["permission_level"],
                )
            )
        return retrieved_docs

    def _build_permission_filters(self, user: UserInfo):
        filters = []

        # Filter by permission_level: user can access documents with permission_level <= user.permission_level
        filters.append(ExactMatchFilter(key="permission_level", value=user.permission_level, operator=FilterOperator.LTE))

        # For user_groups and departments, LlamaIndex's MetadataFilters are primarily designed for
        # single-value metadata. To filter based on whether a user's group/department is *in*
        # a document's list of groups/departments, a more advanced filtering mechanism is needed.
        #
        # Possible approaches for future improvement:
        # 1. Store user_groups and departments as comma-separated strings in metadata and use a 'contains' operator if available.
        # 2. Implement a custom filter within the LlamaIndex VectorStore or QueryEngine.
        # 3. Perform post-retrieval filtering on the retrieved nodes.
        #
        # For the current implementation, we will rely on the permission_level as the primary filter.
        # If "all" is present in the document's user_groups or departments, it means it's accessible to everyone.
        # Otherwise, the user's group/department must be explicitly listed.
        # This logic needs to be applied carefully, potentially requiring a custom filter.

        # Example of how to handle "all" and specific groups/departments (conceptual, not directly supported by MetadataFilters for lists)
        # if user.user_group != "admin": # Assuming admin can see everything
        #     filters.append(ExactMatchFilter(key="user_groups", value=user.user_group, operator=FilterOperator.CONTAINS)) # This operator might not exist for lists
        # if user.department != "all":
        #     filters.append(ExactMatchFilter(key="departments", value=user.department, operator=FilterOperator.CONTAINS)) # This operator might not exist for lists

        return MetadataFilters(filters=filters)
