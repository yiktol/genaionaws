import streamlit as st
import json
import boto3
import math
from numpy import dot
from numpy.linalg import norm

def get_embedding(bedrock, text):
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    input = {
            'inputText': text
        }
    body=json.dumps(input)
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept,contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body['embedding']
    return embedding

def calculate_distance(v1, v2):
    distance = math.dist(v1, v2)
    return distance

def calculate_dot_product(v1, v2):
    similarity = dot(v1, v2)
    return similarity

def calculate_cosine_similarity(v1, v2):
    similarity = dot(v1, v2)/(norm(v1)*norm(v2))
    return similarity

def set_page_config():
    st.set_page_config( 
    page_title="Embedding",  
    page_icon=":rock:",
    layout="wide",
    initial_sidebar_state="expanded",
)
    
def bedrock_runtime_client():
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    return bedrock_runtime