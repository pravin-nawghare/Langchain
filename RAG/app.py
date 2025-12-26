from RAG.components.data_ingestion import DataIngestion
from RAG.components.data_storage import DataStorage

# data ingestion stage
data_ingestion = DataIngestion
data_ingestion.data_ingestion_stage('provide path of folder')

# vectors storing stage
data_storage = DataStorage
data_storage.data_storage_stage(data_ingestion)