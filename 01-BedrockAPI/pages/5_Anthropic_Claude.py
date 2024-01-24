import boto3
import json
import streamlit as st
from helpers import get_models, getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.selectbox('model', get_models('Anthropic'), index=6)
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_k=st.number_input('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 300, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

text, code = st.columns(2)

with text:
    st.title("Anthropic")
    st.write("""Anthropic offers the Claude family of large language models purpose built for conversations, 
            summarization, Q&A, workflow automation, coding and more. 
            Early customers report that Claude is much less likely to produce harmful outputs, 
            easier to converse with, and more steerable - so you can get your desired output with less effort. 
            Claude can also take direction on personality, tone, and behavior.""")

    modelId = model
    accept = 'application/json'
    contentType = 'application/json'

    # Define prompt and model parameters

    with st.form("myform"):
        prompt_data = st.text_input(
            "Ask something:",
            placeholder="Write a python code that list all countries.",
            value = "Write a python code that list all countries."
        )
        submit = st.form_submit_button("Submit")
    body = {"prompt": "Human: " + prompt_data + "\n\nAssistant:",
            "max_tokens_to_sample": max_tokens_to_sample, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31"}

    if prompt_data and submit:
        body = json.dumps(body) # Encode body as JSON string


        #Invoke the model
        response = bedrock_runtime.invoke_model(body=body.encode('utf-8'), # Encode to bytes
                                        modelId=modelId, 
                                        accept=accept, 
                                        contentType=contentType)

        response_body = json.loads(response.get('body').read())
        print(response_body.get('completion'))


        st.write("### Answer")
        st.write(response_body.get('completion'))

with code:

    code = f'''
import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',    
)

body = {json.dumps(body,indent=4)}

modelId = '{model}' 
accept = 'application/json'
contentType = 'application/json'

#Invoke the model
response = bedrock_runtime.invoke_model(
    body=body.encode('utf-8'),
    modelId=modelId, 
    accept=accept, 
    contentType=contentType)

response_body = json.loads(response.get('body').read())

print(response_body.get('completion'))
'''
    
    st.code(code,language="python")