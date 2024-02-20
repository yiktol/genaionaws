import boto3
import json
from helpers import get_models, set_page_config
import streamlit as st

set_page_config()

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.selectbox('model', get_models('AI21'),index=4)
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.7, step = 0.1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 300, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

modelId = model
accept = 'application/json'
contentType = 'application/json'

text, code = st.columns(2)

with text:
   
  st.title("AI21")
  st.write("AI21's Jurassic family of leading LLMs to build generative AI-driven applications and services leveraging existing organizational data. Jurassic supports cross-industry use cases including long and short-form text generation, contextual question answering, summarization, and classification. Designed to follow natural language instructions, Jurassic is trained on a massive corpus of web text and supports six languages in addition to English. ")


  # Define prompt and model parameters

  with st.form("myform"):
    prompt_data = st.text_area(
        "Enter your prompt here:",
        height = 50,
        placeholder="Write me an invitaion letter for my wedding.",
        value = "Write me an invitation letter for my wedding."
    )
    submit = st.form_submit_button("Submit")

  body = {
    "prompt": prompt_data,
    "maxTokens": max_tokens_to_sample,
    "temperature": temperature,
    "topP": top_p,
    "stopSequences": [],
    "countPenalty": {
      "scale": 0
    },
    "presencePenalty": {
      "scale": 0    
    },
    "frequencyPenalty": {
      "scale": 0
    }
  }
  if prompt_data and submit:

    #Invoke the model
    response = bedrock_runtime.invoke_model(body=json.dumps(body), modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    #print(response_body.get('completions')[0].get('data').get('text'))

    st.write("### Answer")
    st.info(response_body.get('completions')[0].get('data').get('text'))
    
  
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
    body=json.dumps(body), 
    modelId=modelId, 
    accept=accept, 
    contentType=contentType
    )

response_body = json.loads(response.get('body').read())

print(response_body.get('completions')[0].get('data').get('text'))
'''

  st.code(code,language="python")