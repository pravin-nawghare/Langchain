from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, CSVLoader, TextLoader, DirectoryLoader
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Any, Optional
import numpy as np

class DataIngestion:
    def __init__(self):
        #self.path = path
        print('Data Ingestion process started')
    
    def _read_data(self,path_of_folder:str) -> List[Any]:
        print('Reading the folder path')
        documents = []
        folder_path = Path(path_of_folder).resolve()
        print(f"Data path: {folder_path}")

        if folder_path == '':
            print(f"Path not provided")
            return None
        
        # For pdf files
        pdf_files = list(folder_path.glob(''))
        print(f"Found {len(pdf_files)} pdf files: {[str(file) for file in pdf_files]}")

        for file in pdf_files:
            print(f'Loading file: {file}')
            try:
                loader = PyPDFLoader(str(file))
                load_file = loader.load()
                print(f"Loaded {len(load_file)} documents from {file}")
                documents.extend(load_file)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        # loader = DirectoryLoader(
        #     path= folder_path,
        #     glob = '*.pdf', # only pdf files from root directory. '**/*' -> all files.
        #     loader_cls= PyPDFLoader # provide what type of loader is like text, pdf, etc
        # )
        return documents


    def data_ingestion_stage(self,path:str):
        document = self._read_data(path)
        return document

