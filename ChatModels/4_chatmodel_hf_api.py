from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

#login()
load_dotenv()

# These models will work
# MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL = "deepseek-ai/DeepSeek-R1"

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.3-70B-Instruct",
                          task="text-generation",
                          )
model = ChatHuggingFace(llm=llm)

# Alernate way
# messages = [
#     ("ai", "You are a helpful assistant."),
#     ("user", "What is the capital of India?"),
# ]
# result = model.invoke(messages)

result = model.invoke("What is the capital of India")

print(result)

