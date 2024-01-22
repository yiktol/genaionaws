import boto3
import json
import streamlit as st
from helpers import get_models


st.set_page_config( 
    page_title="Cohere",
    page_icon=":robot",
    layout="wide",
    initial_sidebar_state="expanded",
)


#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        model = st.selectbox('model', get_models('Cohere'), index=1)
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.75, step = 0.1)
        top_k=st.number_input('top_k',min_value = 0, max_value = 300, value = 0, step = 1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 400, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 


text, code = st.columns(2)

with text:
    st.title('Cohere')
    st.write('Cohere models are text generation models for business use cases. Cohere models are trained on data that supports reliable business applications, like text generation, summarization, copywriting, dialogue, extraction, and question answering.')

    with st.form("myform"):
        prompt_data = st.text_input(
            "Enter your prompt here:",
            placeholder="Write me a poem for my beautiful wife.",
            value = "Write me a poem for my beautiful wife.",
        )
        submit = st.form_submit_button("Submit")

    modelId = model
    accept = 'application/json'
    contentType = 'application/json'

    if prompt_data and submit:
        body = {
            "prompt": prompt_data,
            "max_tokens": max_tokens_to_sample,
            "temperature": temperature,
            "p": top_p,
            "k": top_k,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }

        body = json.dumps(body).encode('utf-8')

        #Invoke the model
        response = bedrock_runtime.invoke_model(body=body,
                                        modelId=modelId, 
                                        accept=accept, 
                                        contentType=contentType)

        response_body = json.loads(response.get('body').read())

        print(response_body['generations'][0]['text'])

        if response_body:
            st.write("### Answer")
            st.write(response_body['generations'][0]['text'])


with code:

    code = '''
        import boto3
        import json

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1', 
        )

        prompt_data = "Write me a poem for my beautiful wife."

        body = {
            "prompt": prompt_data,
            "max_tokens": 400,
            "temperature": 0.75,
            "p": 0.01,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }

        modelId = 'cohere.command-text-v14' 
        accept = 'application/json'
        contentType = 'application/json'

        body = json.dumps(body).encode('utf-8')

        #Invoke the model
        response = bedrock_runtime.invoke_model(body=body,
                                        modelId=modelId, 
                                        accept=accept, 
                                        contentType=contentType)

        response_body = json.loads(response.get('body').read())

        print(response_body['generations'][0]['text'])
        '''
    
    st.code(code,language="python")