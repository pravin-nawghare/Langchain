from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('path of file')
docs = loader.load()
# every page in the provided pdf will become a seperate document and each of the document combine will become the 
# length of the documnet. Any document individually can be accessed.

print(docs)