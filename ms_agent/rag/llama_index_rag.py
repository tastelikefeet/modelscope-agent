import os
from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage
)
from modelscope import snapshot_download
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from omegaconf import DictConfig
from .base import RAG


class LlamaIndexRAG(RAG):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self._validate_config(config)
        self.embedding_model = config.rag.embedding
        self.chunk_size = getattr(config.rag, 'chunk_size', 512)
        self.chunk_overlap = getattr(config.rag, 'chunk_overlap', 50)
        self.retrieve_only = getattr(config.rag, 'retrieve_only', False)
        self.storage_dir = getattr(config.rag, 'storage_dir', './llama_index')

        self._setup_embedding_model(config)

        # Set node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # If retrieve only, don't set LLM
        if self.retrieve_only:
            Settings.llm = None

        self.index = None
        self.query_engine = None

    def _validate_config(self, config: DictConfig):
        """Validate configuration parameters"""
        if not hasattr(config, 'rag') or not hasattr(config.rag, 'embedding'):
            raise ValueError("Missing rag.embedding parameter in configuration")

        chunk_size = getattr(config.rag, 'chunk_size', 512)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

    def _setup_embedding_model(self, config: DictConfig):
        try:
            use_hf = getattr(config, 'use_huggingface', False)
            if not use_hf:
                self.embedding_model = snapshot_download(self.embedding_model)

            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model,
                device='cpu'
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    async def add_documents(self, documents: List[str]) -> bool:
        if not documents:
            raise ValueError("Document list cannot be empty")

        try:
            docs = [Document(text=doc) for doc in documents]
            self.index = VectorStoreIndex.from_documents(docs)
            if not self.retrieve_only:
                self._setup_query_engine()

            return True

        except Exception as e:
            return False

    async def add_documents_from_files(self, file_paths: List[str]) -> bool:
        if not file_paths:
            raise ValueError("File path list cannot be empty")

        try:
            from llama_index.core.readers import SimpleDirectoryReader

            documents = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    continue

                try:
                    if os.path.isfile(file_path):
                        reader = SimpleDirectoryReader(input_files=[file_path])
                    elif os.path.isdir(file_path):
                        reader = SimpleDirectoryReader(input_dir=file_path)
                    else:
                        continue

                    docs = reader.load_data()
                    documents.extend(docs)

                except Exception as e:
                    continue

            if not documents:
                return False

            self.index = VectorStoreIndex.from_documents(documents)

            if not self.retrieve_only:
                self._setup_query_engine()

            return True

        except Exception as e:
            return False

    def _setup_query_engine(self):
        if self.index is None:
            return

        try:
            # Check if LLM is set
            if Settings.llm is None and not self.retrieve_only:
                return

            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )

        except Exception as e:
            pass

    async def _retrieve(self,
                       query: str,
                       limit: int = 5,
                       score_threshold: float = 0.0,
                       **filters) -> List[dict]:
        if self.index is None:
            return []

        if not query.strip():
            return []

        try:
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=limit
            )

            nodes = retriever.retrieve(query)

            # Apply score filtering
            results = []
            for node in nodes:
                if node.score >= score_threshold:
                    results.append({
                        'text': node.node.text,
                        'score': float(node.score),
                        'metadata': node.node.metadata,
                        'node_id': node.node.node_id
                    })

            return results

        except Exception as e:
            return []

    async def retrieve(self,
                              query: str,
                              limit: int = 5,
                              score_threshold: float = 0.0,
                              **filters) -> List[dict]:
        if self.retrieve_only:
            return await self._retrieve(query, limit, score_threshold, **filters)

        if self.index is None or Settings.llm is None:
            return []

        try:
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=limit
            )

            postprocessor = SimilarityPostprocessor(
                similarity_cutoff=score_threshold
            )

            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[postprocessor]
            )

            response = query_engine.query(query)

            results = []
            for node in response.source_nodes:
                results.append({
                    'text': node.node.text,
                    'score': float(node.score),
                    'metadata': node.node.metadata,
                    'node_id': node.node.node_id
                })

            return results

        except Exception as e:
            return []

    async def hybrid_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Hybrid retrieval: Vector retrieval + BM25"""
        if self.index is None:
            return []

        try:
            # Try to import BM25 related modules
            try:
                from llama_index.retrievers.bm25 import BM25Retriever
                from llama_index.core.retrievers import QueryFusionRetriever
                bm25_available = True
            except ImportError:
                bm25_available = False

            # Vector retriever
            vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )

            if not bm25_available:
                # Use vector retrieval only
                nodes = vector_retriever.retrieve(query)
            else:
                # Use hybrid retrieval
                try:
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore=self.index.docstore,
                        similarity_top_k=top_k
                    )

                    fusion_retriever = QueryFusionRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        similarity_top_k=top_k,
                        num_queries=1
                    )

                    nodes = fusion_retriever.retrieve(query)

                except Exception as e:
                    nodes = vector_retriever.retrieve(query)

            results = []
            for node in nodes:
                results.append({
                    'text': node.node.text,
                    'score': float(node.score),
                    'metadata': node.node.metadata,
                    'node_id': node.node.node_id
                })

            return results

        except Exception as e:
            return []

    def query(self, query: str) -> str:
        if self.query_engine is None:
            if self.retrieve_only:
                raise ValueError("Current mode is retrieve only, question answering not supported")
            else:
                raise ValueError("Query engine not initialized, please add documents and set LLM first")

        try:
            response = self.query_engine.query(query)
            return str(response)
        except Exception as e:
            return f"Query failed: {e}"

    def save_index(self, persist_dir: Optional[str] = None):
        """Save index"""
        if self.index is None:
            raise ValueError("No index to save, please add documents first")

        save_dir = persist_dir or self.storage_dir

        try:
            os.makedirs(save_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=save_dir)
        except Exception as e:
            raise

    def load_index(self, persist_dir: Optional[str] = None):
        """Load index"""
        load_dir = persist_dir or self.storage_dir

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Index directory does not exist: {load_dir}")

        try:
            storage_context = StorageContext.from_defaults(persist_dir=load_dir)
            self.index = load_index_from_storage(storage_context)

            # Re-setup query engine
            if not self.retrieve_only:
                self._setup_query_engine()

        except Exception as e:
            raise

    def get_index_info(self) -> dict:
        """Get index information"""
        if self.index is None:
            return {"status": "not_initialized"}

        try:
            doc_count = len(self.index.docstore.docs)
            return {
                "status": "initialized",
                "document_count": doc_count,
                "retrieve_only": self.retrieve_only,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            return {"status": f"error: {e}"}

    def remove_all_documents(self):
        """Remove all documents from the index"""
        try:
            # Clear the index
            self.index = None

            # Clear the query engine
            self.query_engine = None

            # If storage directory exists, optionally clean it up
            if hasattr(self, 'storage_dir') and os.path.exists(self.storage_dir):
                import shutil
                try:
                    shutil.rmtree(self.storage_dir)
                    os.makedirs(self.storage_dir, exist_ok=True)
                except Exception as e:
                    # If we can't remove the directory, just log it but don't fail
                    pass

            return True

        except Exception as e:
            return False

    def remove_documents_by_ids(self, node_ids: List[str]) -> bool:
        """Remove specific documents by their node IDs"""
        if self.index is None:
            raise ValueError("No index exists, please add documents first")

        if not node_ids:
            raise ValueError("Node IDs list cannot be empty")

        try:
            # Get current documents
            docstore = self.index.docstore

            # Remove specified nodes
            for node_id in node_ids:
                if node_id in docstore.docs:
                    docstore.delete_document(node_id)

            # Rebuild index with remaining documents
            remaining_docs = list(docstore.docs.values())

            if not remaining_docs:
                # If no documents remain, clear everything
                self.remove_all_documents()
            else:
                # Rebuild index with remaining documents
                self.index = VectorStoreIndex.from_documents(remaining_docs)

                # Re-setup query engine if not in retrieve-only mode
                if not self.retrieve_only:
                    self._setup_query_engine()

            return True

        except Exception as e:
            return False

    def clear_storage(self, persist_dir: Optional[str] = None):
        """Clear the persistent storage directory"""
        clear_dir = persist_dir or self.storage_dir

        if os.path.exists(clear_dir):
            try:
                import shutil
                shutil.rmtree(clear_dir)
                os.makedirs(clear_dir, exist_ok=True)
                return True
            except Exception as e:
                return False

        return True
