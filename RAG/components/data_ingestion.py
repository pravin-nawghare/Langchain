from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, CSVLoader, TextLoader, DirectoryLoader
from pathlib import Path
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Any, Optional
import numpy as np

class DataIngestion:
    def __init__(self):
        #self.path = path
        print('Data Ingestion process started')
    
    def _read_data(self,path_of_folder) -> List:
        print('Reading the folder path')
        folder_path = Path(path_of_folder)
        if folder_path == '':
            print(f"Path not provided")
            return None
        loader = DirectoryLoader(
            path= folder_path,
            glob = '*.pdf', # provide pattern to load file. for text or csv or pdf -> give the respected pattern
            loader_cls= PyPDFLoader # provide what type of loader is like text, pdf, etc
        )
        load_file = loader.load() # after try lazy_load()
        return load_file

    def _create_chunks(self,document):
        if document == None:
            print("No document provided for chunking")
            return None
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 20,
                chunk_overlap = 0,
                separators= ['\n\n', '\n', ' ', '']
            )
        text_chunks = text_splitter.split_documents(document)
        return text_chunks

    def _create_embeddings(self,chunks) -> np.array:
        if chunks == None:
            print("No text chunks were provided for embedding")
            return None
        embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        embeddings = embedding_model.encode(chunks)
        return embeddings

    def data_ingestion_stage(self,path:str):
        data = self._read_data(path)
        document = self._create_chunks(data)
        embeddings = self._create_embeddings(document)
        return embeddings

