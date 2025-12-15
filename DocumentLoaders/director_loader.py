# loads pdf files from a folder/directory
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# directory loader works well with any type of loader
loader = DirectoryLoader(
    path= 'path of folder',
    glob = '*.pdf', # provide pattern to load file. for text or csv or pdf -> give the respected pattern
    loader_cls= PyPDFLoader # provide what type of loader is like text, pdf, etc
)

docs = loader.load()
print(docs)

'''
Load vs LazyLoad
- loads everything at once                    | loads on demand
- return: a list of document obj              | a generator of document objects
- loads all documents immediately into memory | loaded one at a time
- uses:
1. small no. of documents          | dealing with large no. of documents
2. want everything loaded upfront  | want to stream processing(chunking, embedding) without using lots of memory
'''

