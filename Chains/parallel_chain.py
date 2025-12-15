from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel # for parallel logic

# Flow of code
# document --> 1st branch will create notes --> 2nd branch will create quiz simualtenously --> combine output to # # user

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct", #"mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.3-70B-Instruct",
    task = "text-generation"
)

# Two models for parallel operation
model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

text = """
1. Chains: Chains define sequences of actions, where each step can involve querying an LLM, manipulating data or interacting with external tools. There are two types:
 1. Simple Chains: A single LLM invocation.
 2. Multi-step Chains: Multiple LLMs or actions combined, where each step can take the output from the previous step.

2. Prompt Management: LangChain facilitates managing and customizing prompts passed to the LLM. Developers can use PromptTemplates to define how inputs and outputs are formatted before being passed to the model. It also simplifies tasks like handling dynamic variables and prompt engineering, making it easier to control the LLM's behavior.

3. Agents: Agents are autonomous systems within LangChain that take actions based on input data. They can call external APIs or query databases dynamically, making decisions based on the situation. These agents leverage LLMs for decision-making, allowing them to respond intelligently to changing input.

4. Vector Database: LangChain integrates with a vector database which is used to store and search high-dimensional vector representations of data. This is important for performing similarity searches, where the LLM converts a query into a vector and compares it against the vectors in the database to retrieve relevant information.

Vector database plays a key role in tasks like document retrieval, knowledge base integration or context-based search providing the model with dynamic, real-time data to enhance responses.

5. Models: LangChain is model-agnostic meaning it can integrate with different LLMs such as OpenAI's GPT, Hugging Face models, DeepSeek R1 and more. This flexibility allows developers to choose the best model for their use case while benefiting from LangChain’s architecture.

6. Memory Management: LangChain supports memory management allowing the LLM to "remember" context from previous interactions. This is especially useful for creating conversational agents that need context across multiple inputs. The memory allows the model to handle sequential conversations, keeping track of prior exchanges to ensure the system responds appropriately.
 """
# Prompt to create notes
prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

# Prompt to create quiz
prompt2 = PromptTemplate(
    template = 'Generate a short question from the following text \n {text}',
    input_variables=['text']
)

# Prompt for final output
prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

# Parallel + Merge = Final parallel chain

# creating a parallel chain
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser, # first branch
    'quiz': prompt2 | model2 | parser   # second branch
})

# creating a merge chain/ sequential chain
merge_chain = prompt3 | model1 | parser # final output. Using model1 here, can be any model

# Final chain
chain = parallel_chain | merge_chain
result = chain.invoke({'text':text})

print(result)
# chain.get_graph().print_ascii()

# with open('Langchain Notes and Quiz.txt','w') as file:
#     file.write(result)