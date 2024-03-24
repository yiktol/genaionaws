import boto3

def client(region='us-east-1'):
  return boto3.client(
    service_name='bedrock',
    region_name=region
  )
  
  
def runtime_client(region='us-east-1'):
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=region, 
    )
    return bedrock_runtime  

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

def get_models(provider,region='us-east-1'):
  bedrock = client(region=region)
  all_models = list_available_models(bedrock) 
  models = filter_models_by_provider(all_models, provider)
  return models 


def getmodelId(providername):
    model_mapping = {
        "Amazon" : "amazon.titan-tg1-large",
        "Anthropic" : "anthropic.claude-v2:1",
        "AI21" : "ai21.j2-ultra-v1",
        'Cohere': "cohere.command-text-v14",
        'Meta': "meta.llama2-70b-chat-v1",
        "Mistral": "mistral.mixtral-8x7b-instruct-v0:1",
        "Stability AI": "stability.stable-diffusion-xl-v1"
    }
    
    return model_mapping[providername]

def getmodel_index(providername):
    
    default_model = getmodelId(providername)
    
    idx = getmodelIds(providername).index(default_model)
    
    return idx

def getmodelIds(providername):
    models =[]
    bedrock = client()
    available_models = bedrock.list_foundation_models()
    
    for model in available_models['modelSummaries']:
        if providername in model['providerName']:
            models.append(model['modelId'])
            
    return models

