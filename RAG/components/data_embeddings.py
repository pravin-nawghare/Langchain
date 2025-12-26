from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Any

class DataEmbedding:
    def __init__(self,chunk_size:int = 1000, chunk_overlap:int = 200, model:str = 'all-MiniLM-L6-v2'):
        self.chunk_size = chunk_size
        self.chunk_ooverlap = chunk_overlap
        self.model_name = SentenceTransformer(model)
        print(f"Loading embedding model: {self.model_name}")

    def _create_chunks(self,document:List[Any]) -> List[Any]:
        if document == None:
            print("No document provided for chunking")
            return None
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_ooverlap,
                separators= ['\n\n', '\n', ' ', '']
            )
        text_chunks = text_splitter.split_documents(document)
        print(f"Split {len(document)} documents into {len(text_chunks)} chunks")
        return text_chunks

    def _create_embeddings(self,chunks:List[Any]) -> np.ndarray:
        if chunks == None:
            print("No text chunks were provided for embedding")
            return None
        
        # Initialize the embedding model
        embedding_model = self.model_name
        text = [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(text)} chunk")
        embeddings = embedding_model.encode(text, show_progress_bar=True)

        return embeddings

    def data_storage_stage(self,documnet):
        text_chunks = self._create_chunks(documnet)
        embedded_text = self._create_embeddings(text_chunks)
        return embedded_text