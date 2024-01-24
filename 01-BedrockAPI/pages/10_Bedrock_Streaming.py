import boto3
import json
import streamlit as st
from helpers import get_models, getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.selectbox('model', get_models('Amazon'))
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.7, step = 0.1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 1000, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

model_id = model
accept = 'application/json' 
content_type = 'application/json'

text_gen_config = {
    "maxTokenCount": max_tokens_to_sample,
    "stopSequences": [], 
    "temperature": temperature,
    "topP": top_p
}


text, code = st.columns(2)

with text:
    st.title('Bedrock Streaming')
    st.write("""Invoke the specified Amazon Bedrock model to run inference using the input provided. Return the response in a stream. To find out if a model supports streaming, call GetFoundationModel and check the responseStreamingSupported field in the response.""")


    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = 50,
            placeholder="Write an essay about why someone should drink coffee",
            value = "Write an essay about why someone should drink coffee"  # Set default value
        )
        submit = st.form_submit_button("Submit")

        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig": text_gen_config  
        })


    if prompt_data and submit:
        
        #invoke the model with a streamed response 
        response = bedrock_runtime.invoke_model_with_response_stream(
            body=body, 
            modelId=model_id, 
            accept=accept, 
            contentType=content_type
        )

        st.write("### Answer")
        for event in response['body']:
            data = json.loads(event['chunk']['bytes'])
            #print(data['outputText'])
            st.info(data['outputText'])
            


with code:

    code =f'''
        import boto3
        import json

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',          
            )

        text_gen_config = {{
            "maxTokenCount": {max_tokens_to_sample},
            "stopSequences": [], 
            "temperature": {temperature},
            "topP": {top_p}
            }}

        body = json.dumps({{
            "inputText": {prompt_data},
            "textGenerationConfig": text_gen_config  
            }})

        model_id = '{model}'
        accept = 'application/json' 
        content_type = 'application/json'

        #invoke the model with a streamed response 
        response = bedrock_runtime.invoke_model_with_response_stream(
            body=body, 
            modelId=model_id, 
            accept=accept, 
            contentType=content_type
            )

        for event in response['body']:
            data = json.loads(event['chunk']['bytes'])
            print(data['outputText'])
        '''
    
    st.code(code,language="python")

        
    