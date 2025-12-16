class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerChurnException(e,sys)
    def export_data_into_feature_store(self)->DataFrame:
        try:
            logging.info(f"Exporting data from mongodb")
            customer_data = CustomerChurnData()
            dataframe = customer_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path= os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        
        except Exception as e:
            raise CustomerChurnException(e, sys)

data_ingestion = DataIngestion()
data_ingestion.export_data_into_feature_store()