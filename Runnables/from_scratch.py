import random

class scratch_llm:

    def __init__(self):
        # print('LLM created')
        pass

    def predict(self, prompt):
        response_list = [
            'Delhi is in India',
            'Cricket is most followed in India',
            'Gold rates are rising'
        ]

        return {'response':random.choice(response_list)}
    
llm = scratch_llm()
# response = llm.predict('What means what?')

#  print(response)

class scratch_prompt_template:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)
    
template = scratch_prompt_template(
    template='Write a {length} essay on {topic}',
    input_variables=['topic']
)
result = template.format({'length':'short', 'topic':'cricket'})
# print(result)

# Connecting both classes
output = llm.predict(result)
print(output)