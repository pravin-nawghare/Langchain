from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# with TypedDict --> no validation at the runtime
class Review(TypedDict): # to get output in a structured format, the format is defined
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review) # it tells llm to give output in structured format

result = structured_model.invoke("review")

print(result)
print(result['summary'])
print(result['sentiment'])

# with annotated --> to tell llm what ouptut is needed if it is unable to understand the format/confused about the output format
class REview(TypedDict):
    key_themes: Annotated[list[str], "Write down all key themes in the review "]
    summary: Annotated[str, "A brief summary of the product"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]
#   sentiment: Annotated[Literal['pos','neg'], "Return sentiment of the review either negative, positive or neutral"]    
#   with literal only specified words will be used, instead of it's own words
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
#   optional tells if available then show, otherwise leave it

# it tells llm to give output in structured format
structured_model_annotated = model.with_structured_output(REview) 

result = structured_model_annotated.invoke("review")

print(result)
