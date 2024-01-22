import boto3

def get_bedrock_client():
  return boto3.client(
    service_name='bedrock',
    region_name='us-east-1'
  )

def list_available_models(bedrock_client):
  return bedrock_client.list_foundation_models()

def filter_models_by_provider(all_models, provider):
  active_models = filter_active_models(all_models)
  matching_models = []
  for model in active_models:
    if provider in model['providerName']:
      matching_models.append(model['modelId'])
  return matching_models

def filter_active_models(all_models):
  active_models = []
  for model in all_models['modelSummaries']:
    if 'ACTIVE' in model['modelLifecycle']['status']:
      active_models.append(model)
  return active_models

def get_models(provider):
  bedrock = get_bedrock_client()
  all_models = list_available_models(bedrock) 
  models = filter_models_by_provider(all_models, provider)
  return models 


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
    page_title="Amazon Bedrock API",  
    page_icon=":amazon:",
    layout="wide",
    initial_sidebar_state="expanded",
)
    
def bedrock_runtime_client():
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    return bedrock_runtime