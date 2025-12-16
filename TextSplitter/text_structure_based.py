# MOst used. Text structure inherently have a hirerarchy, using that text splitting is done.
# seperator like - \n\n, \n, ' ', ''  helps in text splitting
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

text = '''
I am in Nagpur
It is known as Orange City

I have spent 1 month here
When winter was on
'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(chunks)
'''Output --> 
['I am in Nagpur', 'It is known as', 'Orange City', 'I have spent 1', 'month here', 'When winter was on']
'''