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

def call_bedrock(modelId, prompt_data):
    if 'amazon' in modelId:
        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig":
            {
                "maxTokenCount":4096,
                "stopSequences":[],
                "temperature":0,
                "topP":0.9
            }
        })
        #modelId = 'amazon.titan-tg1-large'
    elif 'anthropic' in modelId:
        body = json.dumps({
            "prompt": prompt_data,
            "max_tokens_to_sample": 4096,
            "stop_sequences":[],
            "temperature":0,
            "top_p":0.9
        })
        #modelId = 'anthropic.claude-instant-v1'
    elif 'ai21' in modelId:
        body = json.dumps({
            "prompt": prompt_data,
            "maxTokens":4096,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
                    })
        #modelId = 'ai21.j2-grande-instruct'
    elif 'stability' in modelId:
        body = json.dumps({"text_prompts":[{"text":prompt_data}]}) 
        #modelId = 'stability.stable-diffusion-xl'
    else:
        print('Parameter model must be one of titan, claude, j2, or sd')
        return
    accept = 'application/json'
    contentType = 'application/json'

    before = datetime.now()
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    latency = (datetime.now() - before)
    response_body = json.loads(response.get('body').read())

    if 'amazon' in modelId:
        response = response_body.get('results')[0].get('outputText')
    elif 'anthropic' in modelId:
        response = response_body.get('completion')
    elif 'ai21' in modelId:
        response = response_body.get('completions')[0].get('data').get('text')

    #Add interaction to the local CSV file...
    #column_name = ["timestamp", "modelId", "prompt", "response", "latency"] #The name of the columns
    #data = [datetime.now(), modelId, prompt_data, response, latency] #the data
    #with open('./prompt-data/prompt-data.csv', 'a') as f:
    #    writer = csv.writer(f)
    #    #writer.writerow(column_name)
    #    writer.writerow(data)
    
    return response, latency


def getmodelparams(providername):
    model_mapping = {
        "Amazon" : {
            "maxTokenCount": 4096,
            "stopSequences": [], 
            "temperature": 0.5,
            "topP": 0.9
            },
        "Antropic" : {
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
        "Antropic" : "anthropic.claude-v2:1",
        "AI21" : "ai21.j2-ultra-v1"
    }
    
    return model_mapping[providername]

def set_page_config():
    import streamlit as st
    st.set_page_config( 
    page_title="Prompt Engineering",  
    page_icon=":cloud:",
    layout="wide",
    initial_sidebar_state="expanded",
)
    
def bedrock_runtime_client():
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    return bedrock_runtime