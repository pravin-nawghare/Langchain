from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Flow of code
# topic --> llm --> detailed report --> llm --> summary

load_dotenv()

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.3-70B-Instruct", #"google-gemma-2-2b-it" --> didn't work here
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = 'Generate a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = 'Generate 5 pointer summary from following text \n {text}',
    input_variables=['text']
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'India'})

# print(result)
chain.get_graph().print_ascii()