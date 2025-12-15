from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, EmailStr, Field

load_dotenv

model = ChatOpenAI()

# schema
# insert json schema here
Review = dict()
structured_output = model.with_structured_output(Review)

result = structured_output.invoke("Review")

print(result)
print(result.name)