import boto3 
import requests
import json
import streamlit as st
from requests_aws4auth import AWS4Auth 

st.set_page_config( 
    page_title="API",
    page_icon=":robot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Text Generation - Amazon Titan Text")

with st.sidebar:
    st.markdown(
        """
        ### Sample Prompt:
        1. Why is the sky blue?
        2. Who invented the lightbulb?
        
        """
    )
    "[View the source code](https://t.yikyakyuk.com/genai/scripts/bedrock_api.py)"
    
    
session = boto3.Session(profile_name='default',region_name='us-east-1')

credentials = session.get_credentials()

model_id = "amazon.titan-text-lite-v1" #change depending on your model of choice

endpoint = f'https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/invoke'


with st.form("myform"):
  prompt = st.text_input(
      "Ask something:",
      placeholder="Create a recipe for chicken masala."
  )
  submit = st.form_submit_button("Submit")

if prompt and submit:
  payload = {
    'inputText': prompt,
    'textGenerationConfig': {
      'maxTokenCount': 512,
      'stopSequences': [],
      'temperature': 0,
      'topP': 0.9
    } 
  }

  signer = AWS4Auth(credentials.access_key,  
                    credentials.secret_key,
                    'us-east-1', 'bedrock') 
                    
  response = requests.post(endpoint, json=payload, auth=signer)

  print(json.dumps(response.json(), indent=4))
  st.write("### Answer")
  st.write(response.json()['results'][0]['outputText'])