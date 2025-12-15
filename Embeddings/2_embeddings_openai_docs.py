from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embeddings-3-large", dimensions=32)
# more dimensions more contextual meaning gets captured
# small dimension less cost
documents = [
    "Delhi is capital of India",
    "New York is in US",
    "Paris is garbage"
] # a large chunk of text can also be handled at once by this model 
result = embedding.embed_documents(documents)
# 32 dimensionsal vector is returned for each document
print(str(result)) # str to see the vector properly