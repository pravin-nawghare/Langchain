from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embeddings-3-large", dimensions=300)

documents = [

]

query = ""

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score:", score)

# with the help os sentence transformers this can also be done