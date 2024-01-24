import streamlit as st
import boto3
import json
from helpers import get_models, getmodelId, getmodelparams, set_page_config, bedrock_runtime_client
import streamlit as st

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

set_page_config()

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.selectbox('model', get_models('Amazon'))
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 512, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 


text, code = st.columns(2)

with text:

    st.title('Amazon Titan Text')
    st.write("""Titan Text models are generative LLMs for tasks such as summarization, text generation (for example, creating a blog post), classification, open-ended Q&A, and information extraction. They are also trained on many different programming languages as well as rich text format like tables, JSON and csv’s among others.""")

    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area(
        "Enter your prompt here:",
        height = 50,
        placeholder="Create a 3 day itinerary for my visit to Dubai.",
        value="Create a 3 day itinerary for my visit to Dubai.")
        submit = st.form_submit_button("Submit")

    #The Text Generation Configuration are Titans inference parameters 

    text_gen_config = {
        "maxTokenCount": max_tokens_to_sample,
        "stopSequences": [], 
        "temperature": temperature,
        "topP": top_p
    }

    payload = {
            "inputText": str(prompt_data),
            "textGenerationConfig": text_gen_config  
        }
    
    if prompt_data and submit:
        body = json.dumps(payload )

        model_id = model
        accept = 'application/json' 
        content_type = 'application/json'

        # Invoke model 
        response = bedrock_runtime.invoke_model(
            body=body, 
            modelId=model_id, 
            accept=accept, 
            contentType=content_type
        )

        # Print response
        response_body = json.loads(response['body'].read())
        if response_body:
            st.write("### Answer")
            st.info(response_body['results'][0]['outputText'])

with code:
    code = f"""
import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
    )

body = json.dumps({json.dumps(payload,indent=4)})

model_id = '{model}'
accept = 'application/json' 
content_type = 'application/json'

# Invoke model 
response = bedrock_runtime.invoke_model(
    body=body, 
    modelId=model_id, 
    accept=accept, 
    contentType=content_type
)

response_body = json.loads(response['body'].read())
print(response_body['results'][0]['outputText'])
"""
    
    st.code(code,language="python")