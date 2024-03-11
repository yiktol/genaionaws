import streamlit as st
import boto3
import json
from utils import get_models, titan_generic, getmodelparams, set_page_config, bedrock_runtime_client, titan_generic
import utils.helpers as helpers
import streamlit as st

bedrock_runtime = bedrock_runtime_client()

set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.button(label='Reset Session', on_click=form_callback)

dataset = helpers.load_jsonl('utils/titan_text.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6, 0.4])

xcode = f"""import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
    )

body = json.dumps({{
        "inputText": \"{st.session_state["prompt"]}\",
        'textGenerationConfig': {{
            "maxTokenCount": {st.session_state['max_tokens']},
            "stopSequences": [], 
            "temperature": {st.session_state['temperature']},
            "topP": {st.session_state['top_p']}
            }}
        }})
        
response = bedrock_runtime.invoke_model(
    body=body,
    modelId='{st.session_state['model']}', 
    accept='application/json' , 
    contentType='application/json'
)

response_body = json.loads(response['body'].read())
print(response_body['results'][0]['outputText'])
"""

with text:

    st.title('Amazon Titan Text')
    st.write("""Titan Text models are generative LLMs for tasks such as summarization, text generation (for example, creating a blog post), classification, open-ended Q&A, and information extraction. They are also trained on many different programming languages as well as rich text format like tables, JSON and csv’s among others.""")

    with st.expander("See Code"):
        st.code(xcode, language="python")

    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:", height=st.session_state['height'], value=st.session_state["prompt"])
        submit = st.form_submit_button("Submit", type='primary')

    if prompt_data and submit:
        with st.spinner("Generating..."):

            response = helpers.invoke_model(bedrock_runtime, titan_generic(
                prompt_data), st.session_state['model'], max_tokens=st.session_state['max_tokens'], temperature=st.session_state['temperature'], top_p=st.session_state['top_p'])

            st.write("### Answer")
            st.info(response)

with code:
    helpers.tune_parameters('Amazon')

    st.subheader('Prompt Examples:')
    container2 = st.container(border=True)
    with container2:
        helpers.create_tabs(dataset)
