from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda # for conditional logic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

# Flow of code
# Analyze reviews --> check sentiment --> sentiments either +ve or -ve give appropriate reply

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback text")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.3-70B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n{format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

""" Without Pydantic Output Parser
print(classifier_chain.invoke({'feedback':'This is a terrible hair dryer'}))

The sentiment of the feedback text is: Negative.
The word "terrible" has a strongly negative connotation, indicating that the customer is extremely dissatisfied with the hair dryer. 
"""

""" With Pydantic Output Parser
result = classifier_chain.invoke({'feedback':'This is a terrible hair dryer'}).sentiment
print(result)
negative
"""

prompt2 = PromptTemplate(
    template='Write an appropriate response to the positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to the negative feedback \n {feedback}',
    input_variables=['feedback']
)

chain1 = prompt2 | model | parser1
chain2 = prompt3 | model | parser1

branch_chain = RunnableBranch( # pass always tuples which then treated as chains
    (lambda x: x.sentiment == 'positive', chain1), # (condition1, chain1),
    (lambda x: x.sentiment == 'negative', chain2), #(condition2, chain2),
    RunnableLambda(lambda x: "could not find sentiment")  #default_chain
) # for default case RunnableLambda will convert it into chain to execute it

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':'This is a best hair dryer I have ever used'}))
""" 
positive response
Thank you so much for your kind words. We're thrilled to hear that you're happy with our service/product. Your positive feedback means the world to us, and we're grateful for customers like you. If you have any other questions or need assistance in the future, don't hesitate to reach out. We're always here to help. Thanks again for your support!

negative response
I'm so sorry to hear that you're not satisfied. Can you please provide more details about what didn't meet your expectations? Your feedback is invaluable to us, and we'll do our best to make it right and prevent similar issues in the future.
"""