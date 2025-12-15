from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='path.csv')
docs = loader.load()

# every row is a document
print(docs[0])

# custom data loader can also be created for our use specific