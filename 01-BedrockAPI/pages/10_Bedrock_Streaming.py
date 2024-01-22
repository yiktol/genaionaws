import boto3
import json
import streamlit as st



with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.7, step = 0.1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 1000, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

model_id = 'amazon.titan-tg1-large'
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
        prompt_data = st.text_input(
            "Ask something:",
            placeholder="Write an essay about why someone should drink coffee",
            value = "Write an essay about why someone should drink coffee"  # Set default value
        )
        submit = st.form_submit_button("Submit")



    if prompt_data and submit:
        
        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig": text_gen_config  
        })



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
            print(data['outputText'])
            st.write(data['outputText'])

with code:

    code ='''
        import boto3
        import json

        #Create the connection to Bedrock
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1', 
            
        )

        # Define prompt and model parameters
        prompt_data = "Write an essay about why someone should drink coffee"

        text_gen_config = {
            "maxTokenCount": 1000,
            "stopSequences": [], 
            "temperature": 0,
            "topP": 0.9
        }

        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig": text_gen_config  
        })

        model_id = 'amazon.titan-tg1-large'
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

        
    