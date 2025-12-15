from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, EmailStr, Field

load_dotenv

model = ChatOpenAI()

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(description= "Write down all key themes in the review ")
    summary: str = Field(description="A brief summary of the product")
    sentiment: Literal['pos','neg'] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] =  Field(description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(description="Write down all the cons inside a list")
    name: Optional[str] = Field(description="Write the name of the reviewer")
#    email: Optional[str] = EmailStr()    

structured_output = model.with_structured_output(Review)

result = structured_output.invoke("Review")

print(result)
print(result.name)