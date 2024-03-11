import json
import streamlit as st
from utils import get_models, set_page_config, bedrock_runtime_client
import utils.helpers as helpers

set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.button(label='Reset Session', on_click=form_callback)

bedrock_runtime = bedrock_runtime_client()

dataset = helpers.load_jsonl('utils/cohere.jsonl')

helpers.initsessionkeys(dataset[0])
text, code = st.columns([0.6,0.4])

xcode = f'''import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
)

body = json.dumps({{
        "prompt": "{st.session_state['prompt']}",
        "max_tokens": {st.session_state['max_tokens']},
        "temperature": {st.session_state['temperature']},
        "p": {st.session_state['top_p']},
        "k": {st.session_state['top_k']},
        "stop_sequences": [],
        "return_likelihoods": "NONE"
    }})

#Invoke the model
response = bedrock_runtime.invoke_model(
    body=body,
    modelId='{st.session_state['model']}', 
    accept='application/json', 
    contentType='application/json')

response_body = json.loads(response.get('body').read())
print(response_body['generations'][0]['text'])
'''


with text:
    st.title('Cohere')
    st.write('Cohere models are text generation models for business use cases. Cohere models are trained on data that supports reliable business applications, like text generation, summarization, copywriting, dialogue, extraction, and question answering.')

    with st.expander("See Code"): 
        st.code(xcode,language="python")
        
    with st.form("myform"):
        prompt_data = st.text_area("Enter your prompt here:",
            height = st.session_state['height'],
            value = st.session_state["prompt"]
        )
        submit = st.form_submit_button("Submit", type='primary')

    if prompt_data and submit:
        with st.spinner("Generating..."):
            response = helpers.invoke_model(bedrock_runtime, prompt_data, st.session_state['model'], 
                                            max_tokens  = st.session_state['max_tokens'], 
                                            temperature = st.session_state['temperature'], 
                                            top_p = st.session_state['top_p'],
                                            top_k = st.session_state['top_k'])

            st.write("### Answer")
            st.info(response)



with code:

    helpers.tune_parameters('Cohere', index=1)


    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        helpers.create_tabs(dataset)
