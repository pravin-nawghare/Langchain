from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = '''
The Indian Premier League (IPL) is a professional Twenty20 cricket league in India, founded in 2007. It features ten city-based franchise teams and is known as the most popular and richest cricket league in the world. Indian farmers play a crucial role in the country's agriculture, contributing significantly to the economy and food security while facing numerous challenges.

The Global Terrorism Index (GTI) is a report published annually by the Institute for Economics and Peace (IEP), and was developed by IT entrepreneur and IEP's founder Steve Killelea. The index provides a comprehensive summary of the key global trends and patterns in terrorism since 2000. It is an attempt to systematically rank the nations of the world according to terrorist activity. 
'''

text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type='standard_deviation', 
# criteria to check for breaking the text, applied over similarity score of all text splits in the documnet
    breakpoint_threshold_amount=1 # play with it to get better result
# if similarity score goes beyond breakpoint_threshold_amount for this criteria breakpoint_threshold_type, meaning of text is changed is considered and text splitting is done
)

docs = text_splitter.create_documents([text])

print(len(docs))
print(docs)