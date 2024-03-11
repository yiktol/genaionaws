import boto3
import json
import streamlit as st
from utils import claude_generic, set_page_config, bedrock_runtime_client
import utils.helpers as helpers
set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.button(label='Clear Session', on_click=form_callback)

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

dataset = helpers.load_jsonl('utils/anthropic.jsonl')
# print(type(dataset[0].keys()))

helpers.initsessionkeys(dataset[0])
text, code = st.columns([0.6,0.4])

xcode = f'''import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',    
)

body = {{"prompt": \"{st.session_state['prompt']}\",
        "max_tokens_to_sample": {st.session_state['max_tokens']}, 
        "temperature": {st.session_state['temperature']},
        "top_k": {st.session_state['top_k']},
        "top_p": {st.session_state['top_p']},
        "stop_sequences": ["\\n\\nHuman:"],
        "anthropic_version": "bedrock-2023-05-31"}}

modelId = '{st.session_state['model']}' 
accept = 'application/json'
contentType = 'application/json'

#Invoke the model
response = bedrock_runtime.invoke_model(
    body=json.dumps(body.encode('utf-8')),
    modelId=modelId, 
    accept=accept, 
    contentType=contentType)

response_body = json.loads(response.get('body').read())

print(response_body.get('completion'))
'''

with text:
    st.title("Anthropic")
    st.write("""Anthropic offers the Claude family of large language models purpose built for conversations, 
            summarization, Q&A, workflow automation, coding and more. 
            Early customers report that Claude is much less likely to produce harmful outputs, 
            easier to converse with, and more steerable - so you can get your desired output with less effort. 
            Claude can also take direction on personality, tone, and behavior.""")

    with st.expander("See Code"): 
        st.code(xcode,language="python")

    modelId = st.session_state['model']
    accept = 'application/json'
    contentType = 'application/json'

    # Define prompt and model parameters
    prompt_input = "Write a python code that list all countries."
    
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = st.session_state['height'],
            value = st.session_state["prompt"]
        )
        submit = st.form_submit_button("Submit", type='primary')
        
    if prompt_data and submit:
        with st.spinner("Generating..."):
            response = helpers.invoke_model(bedrock_runtime, claude_generic(prompt_data), st.session_state['model'], 
                                            max_tokens  = st.session_state['max_tokens'], 
                                            temperature = st.session_state['temperature'], 
                                            top_p = st.session_state['top_p'],
                                            top_k = st.session_state['top_k'])

            st.write("### Answer")
            st.info(response)

with code:

    helpers.tune_parameters('Anthropic',index=6)
    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        helpers.create_tabs(dataset)