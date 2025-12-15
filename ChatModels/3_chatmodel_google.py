from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-001")
# (model="gemini-1.5-pro")

result = model.invoke("What is the capital of India")

print(result)