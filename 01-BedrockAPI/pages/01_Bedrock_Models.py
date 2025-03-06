import streamlit as st
from utils import set_page_config, bedrock_client

set_page_config()


st.title("Bedrock Models")
st.write("""Amazon Bedrock is a fully managed service that offers a choice of high-performing \
foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, \
and Amazon via a single API, along with a broad set of capabilities you need to build generative AI \
applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment \
with and evaluate top FMs for your use case, privately customize them with your data using techniques such \
as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise \
systems and data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, \
and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services \
you are already familiar with.""")

# with st.sidebar:
#     provider = st.selectbox('Provider', ('Amazon', 'Stability',
#                             'AI21', 'Anthropic', 'Cohere', 'Meta', 'Mistral'),)
#     Region = st.selectbox('Region', ('us-east-1', 'us-west-2',
#                           'ap-southeast-1', 'ap-northeast-1', 'eu-central-1',))

# Create the connection to Bedrock



def get_models(provider):

    bedrock = bedrock_client(region=Region)
    available_models = bedrock.list_foundation_models()
    # print(available_models)

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

def show_code(provider, Region):
    code = f"""import boto3
import json

bedrock = boto3.client(service_name='bedrock', region_name='{Region}')

available_models = bedrock.list_foundation_models()

for model in available_models['modelSummaries']:
if '{provider}' in model['providerName']:
    print(json.dumps(model, indent=4))

    """
    return code

with st.container():
    pycode, region = st.columns([0.7,0.3])    
    
    with region:
        with st.container(border=True):
            
            provider = st.selectbox('Provider', ('Amazon', 'Stability',
                                    'AI21', 'Anthropic', 'Cohere', 'Meta', 'Mistral'),)
            Region = st.selectbox('Region', ('us-east-1', 'us-west-2',
                                'ap-southeast-1', 'ap-southeast-2','ap-northeast-1', 'eu-central-1',))
    with pycode:
        st.code(show_code(provider, Region), language="python")
        
st.table(get_models(provider))
