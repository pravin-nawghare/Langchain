from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Flow of code -->
# IP -> LLM -> OP is text -> LLM -> 5 line summary

llm = HuggingFaceEndpoint(
    repo_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # this llm dosen't support
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt --> detailed report
template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt --> summary in five lines
template2= PromptTemplate(
    template = "Write a 5 line summary on the following text. /n report on {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# Flow --> 
"""
template1: input prompt to the llm
model: llm which will give the first output
parser: llm output is cluttered, which will clean up by parser
template2: input prompt to next llm
model: llm which will give last output
parser: cluttered output will get cleaned by parser. It is the final output.
"""
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'}) # dict is needed with chain.invoke

print(result)