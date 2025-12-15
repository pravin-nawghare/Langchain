from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
# from langchain import a json output parser which have a schema attribute
# no data validation but enforces schema on output

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# create a parser here

template = PromptTemplate(
    template='Give 3 fact above {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
