import io
import streamlit as st
import sys
import json
import boto3
from datetime import date
import boto3
import json

st.markdown("""#### Automatic Reasoning and Tool-use (ART)
Combining CoT prompting and tools in an interleaved manner has shown to be a strong and robust approach to address many tasks with LLMs. These approaches typically require hand-crafting task-specific demonstrations and carefully scripted interleaving of model generations with tool use. Paranjape et al., (2023) propose a new framework that uses a frozen LLM to automatically generate intermediate reasoning steps as a program.

ART works as follows:
- given a new task, it select demonstrations of multi-step reasoning and tool use from a task library
- at test time, it pauses generation whenever external tools are called, and integrate their output before resuming generation           
            
            """)

instruct_mistral7b_id = "mistral.mistral-7b-instruct-v0:2"
instruct_mixtral8x7b_id = "mistral.mixtral-8x7b-instruct-v0:1"

DEFAULT_MODEL = instruct_mixtral8x7b_id

class LLM:
    def __init__(self, model_id):
        self.model_id = model_id
        self.bedrock = boto3.client(service_name="bedrock-runtime",region_name='us-east-1')
        
    def invoke(self, prompt, temperature=0.0, max_tokens=128):
        body = json.dumps({
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt, 
            "stop": ["</s>"]
        })
        response = self.bedrock.invoke_model(
            body=body, 
            modelId=self.model_id)

        response_body = json.loads(response.get("body").read())
        return response_body['outputs'][0]['text']
    
llm = LLM(DEFAULT_MODEL)



def get_current_date() -> str:
    return str(date.today())

import re
def get_ticket_price_by_age(age: str) -> int:
    
    try:
        res = re.findall(r'[\d]*[.][\d]+', age)
        age = round(float(res[0]),0)
    except Exception:
        age = int(re.findall(r'[\d]+', age)[0])
        
    match age:
        case age if age <= 5:
            return 0
        case age if age <= 15:
            return 15
        case age if age <= 30:
            return 30
        case age if age <= 45:
            return 45
        case age if age >45:
            return 20
    



tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get today's date, use this for any questions related to knowing today's date.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price_by_age",
            "description": "Get the ticket price by age.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "age of the person",
                    }
                },
                "required": ["age"],
            },
        },
    }
]


def next_step(response):
    instruction = response[ : response.find('\n- Observation:')]
    lines = instruction[instruction.rfind("Action:"):].split("\n")
    action, action_input = lines[0].split(": ")[1].strip(), lines[1].split(": ")[1].strip()
    func = globals().get(action)
    if action_input == "None":
        observation = func()
    else:
        observation = func(action_input)
    return str(instruction) + '\n- Observation: ' + str(observation) + '\n- Thought:'

import functools

names_to_functions = {
    'get_current_date': functools.partial(get_current_date),
    'get_ticket_price_by_age': functools.partial(get_ticket_price_by_age)
}

def fill_function_calling_template(question, tools):
    query = f'''<s> [INST] You are a useful AI agent. Answer the following Question as \
best you can. You have access to the following tools:

Tools = {[item["function"]["name"] + ": " + item["function"]["description"] for item in tools]}

Use the following format:

### Start
- Question: the input question you must answer
- Thought: explain your reasoning about what to do next
- Action: the action to take, should be one of {[item["function"]["name"] for item in tools]}
- Action Input: the input to the action
- Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times until the final answer is known)
- Thought: I now know the final answer
- Final Answer: the final answer to the original input question

Follow this format and Start! [/INST]

### Start
- Question: {question}
- Thought:'''
    return query


def prompt_box(prompt,key,height):
	with st.form(f"form-{key}"):
		prompt_data = st.text_area(
			":orange[Enter your prompt here:]",
			height = height,
			value=prompt,
			key=key)
		submit = st.form_submit_button("Submit", type="primary")
	
	return submit, prompt_data
		 

question = "I was born on Jan 02, 1999. How much do I need to pay for the ticket?"


# print(query)
col1, col2 = st.columns([0.8,0.2])

with col2:
    st.dataframe({
    "Age": ["<=5", "<=15", "<=30", "<=45", ">45"],
    "Ticket Price": [0, 15, 30, 45, 20]
})

with col1:
    with st.expander("View Open API schema:"):
        st.code(json.dumps(tools, indent=4), language="json")
        
    submit, prompt_data = prompt_box(question ,1,50)
    if submit:
        with st.spinner("Thinking..."):
            query = fill_function_calling_template(prompt_data, tools)
            queries = [query]
            
            response = llm.invoke(
                query,
                temperature=0.0,
                max_tokens=2048,
            )
            # st.write(response)
            # st.write("---")
            response_observation = next_step(response)
            queries.append(response_observation)
            # st.write(''.join(queries[:-1]) + '\033[32m\033[1m' + queries[-1])


            response = llm.invoke(
                ''.join(queries),
                temperature=0.0,
                max_tokens=2048,
            )
            # st.write(response)
            # st.write("---")

            response_observation = next_step(response)
            queries.append(response_observation)
            # print(''.join(queries[:-1]) + '\033[32m\033[1m' + queries[-1])

            response = llm.invoke(
                ''.join(queries),
                temperature=0.0,
                max_tokens=2048,
            )
            st.write("Instructions")
            st.info(query)
            st.write("Answer")
            st.success(queries[1] + queries[2] + response)