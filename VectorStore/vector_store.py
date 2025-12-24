'''
Vector Store - provides storage, retriveal and semantic search functionality.
e.g.- FAISS
Vectore Database - with all functionality of vector store additional functionalities of database like concurrency,
ACID, etc.
e.g.- Pinecone, AstraDB
'''
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Create documents
doc1 = Document(
    page_content="This is a test document about machine learning. Machine Learning is fast growing field in today's world. Those without sound knowledge of maths now can work in this domain",
    metadata =({'source':'Machine Learning'})
)
doc2 = Document(
    page_content="This is a test document about data analysis. Data Analysis is considered as a sub-branch of machine learning. It involves inspecting, cleansing, transforming, and modeling data to discover useful information.",
    metadata =({'source':'Data Analysis'})
)
doc3 = Document(
    page_content="This is a test document about robotics. Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electronic engineering, information engineering, computer science, and others. Robotics deals with the design, construction, operation, and use of robots.",
    metadata =({'source':'Robotics'})
)
doc4 = Document(
    page_content="This is a test document about ethical hacking. Ethical Hacking is the practice of legally breaking into computers and devices to test an organization's defenses. These hackers are often referred to as 'white hats.'",
    metadata =({'source':'Ethical Hacking'})
)
doc5 = Document(
    page_content="This is a test document about devops. Devops is a set of practices that combines software development (Dev) and IT operations (Ops). It aims to shorten the systems development life cycle and provide continuous delivery with high software quality.",
    metadata =({'source':'Devops'})
)

doc = [doc1, doc2, doc3, doc4, doc5]

# Create Embedding 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='./VectorStore/chroma_db',
    collection_name='software_jobs'
)

# add documents
vector_store.add_documents(documents=doc)
 