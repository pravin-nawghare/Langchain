# this can also be connected to models
from langchain_community.document_loaders import TextLoader

loader = TextLoader("hugging face.txt",encoding='utf-8')
docs = loader.load()

# print(docs)
# Output: [Document(metadata={'source': 'hugging face.txt'}, page_content='Hugging face\n\npassword\n@ApiProject2025')]
print(len(docs)) # 1
print(type(docs)) # <class 'list'>
print(docs[0])
'''
page_content='Hugging face

password
@ApiProject2025' metadata={'source': 'hugging face.txt'}
'''
print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>
