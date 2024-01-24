import boto3
import json
import streamlit as st
from helpers import set_page_config

set_page_config()

#Create the connection to Bedrock
bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1', 
    
)
st.title("Bedrock Models")


with st.sidebar:
    provider = st.selectbox(
    'Provider',
    ('Amazon', 'Stability', 'AI21', 'Anthropic','Cohere','Meta'),)

def get_models(provider):

   # Let's see all available Amazon Models
    available_models = bedrock.list_foundation_models()

    models = [{}]


    for each_model in available_models['modelSummaries']:
        if provider in each_model['providerName']:
            model = {}
            model["Name"] = each_model['modelName']
            model["inputModalities"] = each_model['inputModalities']
            model["ouputModalities"] = each_model['outputModalities']
            model["Id"] = each_model['modelId']
            model["ProviderName"] = each_model['providerName']
            models.append(model)
    models.pop(0)

    return models

code = f"""
import boto3
import json

bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')

available_models = bedrock.list_foundation_models()

for model in available_models['modelSummaries']:
  if {provider} in model['modelId']:
    print(json.dumps(model, indent=4))

"""

st.code(code,language="python")

st.table(get_models(provider))