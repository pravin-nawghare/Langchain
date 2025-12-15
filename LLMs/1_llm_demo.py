from langchain_openai import OpenAI # needs api key to run it's llm
from dotenv import load_dotenv # to load env file

load_dotenv() # loading environment file

llm = OpenAI(model="gpt-3.5-turbo-instruct")
# A llm takes input as string or plain text and gives output also as a string

result = llm.invoke("What is a capital of India")

print(result)