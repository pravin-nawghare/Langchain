from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
import numpy as np
from typing import List, Any

class DataChunking:
    def __init__(self,chunk_size:int = 800, chunk_overlap:int = 160):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print("-------------------Data Chunking Stage started-------------------")

    def _create_chunks(self,document:List[Any]) -> List[Document]:
        if document == None:
            print("No document provided for chunking")
            return None
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap,
                separators= ['\n\n', '\n', ' ', '']
            )
        
        # Read the document
        try:
            text_chunks = text_splitter.split_documents(document)
            print(f"Split {len(document)} documents into {len(text_chunks)} chunks")
        except Exception as e:
            print(f"Error during text chunking: {e}")
            raise

        return text_chunks

    def data_chunking_stage(self,documnet):
        text_chunks = self._create_chunks(documnet)
        return text_chunks
    
