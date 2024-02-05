import json
import boto3
from datetime import datetime
from langchain.prompts import PromptTemplate

#Create the connection to Bedrock
bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1', 
    
)

def claude_generic(input_prompt):
    prompt = f"""Human: {input_prompt}\n\nAssistant:"""
    return prompt

def titan_generic(input_prompt):
    prompt = f"""User: {input_prompt}\n\nAssistant:"""
    return prompt

def llama2_generic(input_prompt, system_prompt):
    prompt = f"""<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {input_prompt} [/INST]
    """
    return prompt

def getmodelparams(providername):
    model_mapping = {
        "Amazon" : {
            "maxTokenCount": 4096,
            "stopSequences": [], 
            "temperature": 0.5,
            "topP": 0.9
            },
        "Anthropic" : {
            "max_tokens_to_sample": 4096,
            "temperature": 0.9,
            "top_k": 250,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman"],
            },
        "AI21" : {
            "maxTokens": 4096,
            "temperature": 0.5,
            "topP": 0.9,
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
    }
    
    return model_mapping[providername]

def getmodelId(providername):
    model_mapping = {
        "Amazon" : "amazon.titan-tg1-large",
        "Anthropic" : "anthropic.claude-v2:1",
        "AI21" : "ai21.j2-ultra-v1"
    }
    
    return model_mapping[providername]

def set_page_config():
    import streamlit as st
    st.set_page_config( 
    page_title="GenAI Tools",  
    page_icon=":gear:",
    layout="wide",
    initial_sidebar_state="expanded",
)
    
def bedrock_runtime_client():
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    return bedrock_runtime