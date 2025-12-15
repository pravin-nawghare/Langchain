from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()

template = PromptTemplate(
    template="Give 5 interesting facts about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)
chain.get_graph().print_ascii() # it will show the full pipeline