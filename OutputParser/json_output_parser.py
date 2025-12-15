from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Flow of code -->
# IP -> LLM -> OP is text -> LLM -> 5 line summary

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser() # cannot tell schema of output. It is determined by llm only

template =PromptTemplate( # format instruction tells it is getting a JSON object
    template='Give me the name, age and city of a fictional person \n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)  # parser.get_format_instructions() --> giving output to format_instructions which is json type

# without chains
prompt = template.format()

result = model.invoke(prompt)

final_result = parser.parse(result.content) # parsing to get structured output

print(final_result) # output is dict

# with chains
chain = template | model | parser

result = chain.invoke({}) # as no input variables are used but a dict is needed so sending a empty dict

print(result)
