import boto3
import json
import streamlit as st
from helpers import get_models, set_page_config


set_page_config()


#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
)

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.selectbox('model', get_models('Meta'), index=1)
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.7, step = 0.1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 512, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 


modelId = model
accept = 'application/json'
contentType = 'application/json'

text, code = st.columns(2)

with text:

    st.title("Meta")
    st.write("Llama is a family of large language models that uses publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.")


    # Define prompt and model parameters
    with st.form("myform"):
        prompt = st.text_area(
            "Enter your prompt here:",
            height = 50,
            placeholder="Describe the plot of the TV show Breaking Bad.",
            value = "Describe the plot of the TV show Breaking Bad."  # Default value is 'Hello'
        )
        submit = st.form_submit_button("Submit")

        body = { 
            'prompt': prompt,
            'max_gen_len': max_tokens_to_sample,
            'top_p': top_p,
            'temperature': temperature
        }

    if prompt and submit:
        #Invoke the model
        response = bedrock_runtime.invoke_model(body=json.dumps(body).encode('utf-8'), # Encode to bytes
                                        modelId=modelId, 
                                        accept=accept, 
                                        contentType=contentType)

        response_body = json.loads(response.get('body').read().decode('utf-8'))
        print(response_body.get('generation'))

        st.write("### Answer")
        st.info(response_body.get('generation'))

    #We can also call the Meta Llama 2 models via the streaming API
            
    # response = bedrock_runtime.invoke_model_with_response_stream(body=body.encode('utf-8'), # Encode to bytes
    #                                 modelId=modelId, 
    #                                 accept=accept, 
    #                                 contentType=contentType)

    # event_stream = response.get('body')
    
    # st.write("### Answer")
    # for b in iter(event_stream):
    #     bc = b['chunk']['bytes']
    #     gen = json.loads(bc.decode('utf-8'))
    #     line = gen.get('generation')
    #     st.markdown(line)
    #     if '\n' == line:
    #         print('')
    #         continue
    #     print(line, end='')
        
with code:

    code = f'''
import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )

body = {json.dumps(body,indent=4)}

modelId = '{model}'
accept = 'application/json'
contentType = 'application/json'

#Invoke the model
response = bedrock_runtime.invoke_model(
    body=body.encode('utf-8'),
    modelId=modelId, 
    accept=accept, 
    contentType=contentType)

response_body = json.loads(response.get('body').read().decode('utf-8'))

print(response_body.get('generation'))
'''

    st.code(code,language="python")
        
