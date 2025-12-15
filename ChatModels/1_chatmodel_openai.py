from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")
# attributes provided inside
# temperature
# max_completion_tokens --> restricts output tokens

result = model.invoke("What is the capital of India")

print(result.content) # the output contains too many information and to extract only answer we fetch only 'content'
