from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embeddings-3-large", dimensions=32)
# more dimensions more contextual meaning gets captured
# small dimension less cost
result = embedding.embed_query("Delhi is the capital of India")
# 32 dimensionsal vector is return
print(str(result)) # str to see the vector properly