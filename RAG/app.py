from components.data_ingestion import DataIngestion
from components.vector_store import VectorStore
from components.data_chunking import DataChunking

# data ingestion stage
data_ingestion = DataIngestion(path="./RAG/data")
docs = data_ingestion.data_ingestion_stage()
print('-------------------Data Ingestion Stage completed-------------------')

# vectors creation stage
data_chunks = DataChunking()
chunks = data_chunks.data_chunking_stage(docs)
# print(chunks)
print('-------------------Data Chunking Stage completed-------------------')

# vectors storing stage
vector_store = VectorStore(model="sentence-transformers/all-MiniLM-L6-v2")
vector_store_ids = vector_store.embedding_storage_stage(chunks)
print('-------------------Vector Storing Stage completed-------------------')


