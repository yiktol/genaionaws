import boto3
import json
import streamlit as st
from utils import get_models, set_page_config, llama2_generic, bedrock_runtime_client
import utils.helpers as helpers

set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.button(label='Clear Session', on_click=form_callback)

bedrock_runtime = bedrock_runtime_client()

dataset = helpers.load_jsonl('utils/meta.jsonl')

helpers.initsessionkeys(dataset[0])
text, code = st.columns([0.6,0.4])

default_system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
Please ensure that your responses are socially unbiased and positive in nature. \
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
If you don't know the answer to a question, please don't share false information."""

xcode = f'''import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    
default_system_prompt = \"{default_system_prompt}\"

body = json.dumps({{'prompt': \"{st.session_state['prompt']}\",
        'max_gen_len': {st.session_state['max_tokens']},
        'top_p': {st.session_state['top_p']},
        'temperature': {st.session_state['temperature']}
    }})

#Invoke the model
response = bedrock_runtime.invoke_model(
    body=body,
    modelId='{st.session_state['model']}', 
    accept='application/json', 
    contentType='application/json')

response_body = json.loads(response.get('body').read().decode('utf-8'))
print(response_body.get('generation'))
'''



input_prompt = "Can you explain what a transformer is (in a machine learning context)?"

with text:

    st.title("Meta")
    st.write("Llama is a family of large language models that uses publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.")

    with st.expander("See Code"): 
        st.code(xcode,language="python")

    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area("Enter your prompt here:",
            height = st.session_state['height'],
            value = st.session_state["prompt"]
        )
        submit = st.form_submit_button("Submit", type='primary')

        body = { 
            'prompt': llama2_generic(prompt_data,default_system_prompt),
            'max_gen_len': st.session_state['max_tokens'],
            'top_p': st.session_state['top_p'],
            'temperature': st.session_state['temperature']
        }

    if prompt_data and submit:
        with st.spinner("Generating..."):
            response = helpers.invoke_model(bedrock_runtime, prompt_data, st.session_state['model'], 
                                            max_tokens  = st.session_state['max_tokens'], 
                                            temperature = st.session_state['temperature'], 
                                            top_p = st.session_state['top_p'])

            st.write("### Answer")
            st.info(response)



        
with code:


    helpers.tune_parameters('Meta', index=3)


    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        helpers.create_tabs(dataset)
        
