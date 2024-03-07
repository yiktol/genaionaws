import boto3
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
import streamlit as st
import json
from helpers import get_models, getmodelId, getmodelparams, set_page_config, bedrock_runtime_client


#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',   
    )

set_page_config()

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.selectbox('model', get_models('Anthropic'), index=6)
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_k=st.number_input('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 300, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters', type='primary') 


text, code = st.columns(2)

template = 'Human: {task}\n\nAssistant:'
task = """Write an email from Bob, Customer Service Manager, to the customer "John Doe" that provided negative \
feedback on the service provided by our customer support engineer."""

inference_modifier = {
    "max_tokens_to_sample": max_tokens_to_sample,
    "temperature": temperature,
    "top_k": top_k,
    "top_p": top_p,
    "stop_sequences": ["\n\nHuman"], }


def format_prompt():
    prompt = PromptTemplate(
        input_variables=["task"], 
        template=template)

    prompt_query = prompt.format(task=task)
    
    return prompt_query

def call_llm():

    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock_runtime,
        model_kwargs=inference_modifier
        )
    prompt_query = format_prompt()

    response = llm(prompt_query)

    return response



with text:
    st.title("Langchain")
    st.write("LangChain is a framework for developing applications powered by language models.")

    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = 150,
            value = format_prompt()
            )
        submit = st.form_submit_button("Submit")

    if prompt_data and submit:

        response = call_llm()

        #print(response)
        st.write("### Answer")
        st.info(response)
  

with code:

    code = f'''
import boto3
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )

template = "{template}"
task = "{task}"

inference_modifier = {json.dumps(inference_modifier,indent=4)}

def call_llm():

    llm = Bedrock(
        model_id="{model}",
        client=bedrock_runtime,
        model_kwargs=inference_modifier
        )

    prompt = PromptTemplate(
        input_variables=["task"], 
        template=template)
        
    prompt_query = prompt.format(task=task)
    response = llm(prompt_query)
    
    return response

print(call_llm())
'''

    st.code(code,language="python")
