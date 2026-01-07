from typing import Any, List, Tuple, Dict
import uuid
import os
import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self,model:str, chunk_size:int = 1000, chunk_overlap:int = 200 ,persist_dir:str='./RAG/chroma_store',):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model
        self.persist_directory = persist_dir
        os.makedirs(self.persist_directory, exist_ok=True)
        # creating chromadb client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        print('-------------------Vector Store Stage started-------------------')
        print(f"with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, persist_directory={self.persist_directory}")
        print(f"Loading embedding model: {self.model_name}")
    
    def _vectors_storage(self,chunks:List[Document]):
        vector_store = Chroma(
            client=self.client,
            embedding_function= HuggingFaceEmbeddings(model_name=self.model_name),
            collection_name='interview_questions'
        )

        # Initialize the vector store
        try:
            uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            print(f"Generated {len(uuids)} UUIDs for the chunks")
            store_ids = vector_store.add_documents(documents=chunks, ids=uuids) # gives ids
            return store_ids
        except Exception as e:
            print(f"Error while adding documents to vector store: {e}")
            raise

    def embedding_storage_stage(self,chunk:List[Document]):
        ids = self._vectors_storage(chunk)
        return ids