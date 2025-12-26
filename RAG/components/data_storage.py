from langchain_chroma import Chroma
# from RAG.data_ingestion import DataIngestion
from sentence_transformers import SentenceTransformer
import numpy as np

class DataStorage:
    def __init__(self):
        pass

    def _create_embeddings(self,chunks) -> np.array:
        if chunks == None:
            print("No text chunks were provided for embedding")
            return None
        embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        return embedding_model

    def _vectors_storing(embedding_model):
        vector_store = Chroma(
            embedding_function= embedding_model,
            persist_directory='chroma_db',
            collection_name='software_jobs'
        )
        pass

    def data_storage_stage():
        pass