# this can also be connected to models
from langchain_community.document_loaders import TextLoader

loader = TextLoader("hugging face.txt",encoding='utf-8')
docs = loader.load()

# print(docs)
print(len(docs)) # 1
print(type(docs)) # <class 'list'>
print(docs[0])
'''
Output present in original text loader file
'''
print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>
