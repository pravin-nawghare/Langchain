from langchain_community.document_loaders import PyPDFLoader,TextLoader
from pathlib import Path
from typing import List, Any

class DataIngestion:
    def __init__(self,path:str):
        self.path = path
        print('-------------------Data Ingestion process started-------------------')
    
    def _read_data(self) -> List[Any]:
        if self.path == '':
            print(f"Path not provided")
            return None
        
        print('Reading the directory path')
        documents = []
        directory_path = Path(self.path).resolve()
        print(f"Directory path: {directory_path}")
        
        # For pdf files
        pdf_files = list(directory_path.glob('**/*.pdf'))
        print(f"Found {len(pdf_files)} pdf files: {[str(file).split('\\')[-1] for file in pdf_files ]}")

        for file in pdf_files: 
            print(f'Loading file: {[str(file).split('\\')[-1]]}')
            try: 
                loader = PyPDFLoader(str(file))
                load_file = loader.load()
                print(f"Loaded {len(load_file)} documents from {[str(file).split('\\')[-1]]}")
                documents.extend(load_file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                raise

        return documents


    def data_ingestion_stage(self):
        document = self._read_data()
        return document

# data_ingestion = DataIngestion(path="./RAG/data")
# docs = data_ingestion.data_ingestion_stage()
# print(docs[3].page_content)