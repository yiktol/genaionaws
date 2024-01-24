import boto3
import json
import streamlit as st
import numpy as np
from helpers import set_page_config

set_page_config()

#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

model_id = 'amazon.titan-embed-g1-text-02' #look for embeddings in the modelID
accept = 'application/json' 
content_type = 'application/json'

with st.sidebar:
    "Model attributes"
    "Output vector size = 1,536"
    "Max tokens: 8k"
    "modelId: amazon.titan-embed-text-v1"

st.title("Vertorize using Titan Embeddings")

# Define prompt and model parameters
with st.form("myform"):
    prompt_data = st.text_input(
        "Enter a prompt to vectorize", 
        placeholder="Write me a poem about apples",
        value="Write me a poem about apples")
    submitted = st.form_submit_button("Vectorize")

if prompt_data and submitted:
    body = json.dumps({
        "inputText": prompt_data,
    })


    # Invoke model 
    response = bedrock_runtime.invoke_model(
        body=body, 
        modelId=model_id, 
        accept=accept, 
        contentType=content_type
    )

    # Print response
    response_body = json.loads(response['body'].read())
    embedding = response_body.get('embedding')

    
    
    # def print_matrix_numpy(matrix):
    #     # Convert the list to a NumPy array
    #     np_matrix = np.array(matrix)
    #     # Print the NumPy array
    #     return np_matrix

    #Print the Embedding
    #print(embedding)
    st.write(embedding)
    #print(embedding)