import streamlit as st
import jsonlines
import json
import utils
import base64
from io import BytesIO
from random import randint
from jinja2 import Environment, FileSystemLoader
from utils import bedrock_runtime_client

def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d

def initsessionkeys(dataset):
    for key in dataset.keys():
        if key not in st.session_state:
            st.session_state[key] = dataset[key]
    return st.session_state

def update_options(dataset,item_num):
    for key in dataset[item_num]:
        st.session_state[key] = dataset[item_num][key]

def load_options(dataset,item_num):    
    st.write("Prompt:",dataset[item_num]["prompt"])
    st.write("Negative Prompt:", dataset[item_num]["negative_prompt"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(dataset,item_num))  

def update_parameters(**args):
    if 'temperature' in args:
        st.session_state['temperature'] = args['temperature']
    if 'top_p' in args:
        st.session_state['top_p'] = args['top_p']
    if 'top_k' in args:
        st.session_state['top_k'] = args['top_k']
    if 'model' in args:
        st.session_state['model'] = args['model']
    if 'max_tokens' in args:
        st.session_state['max_tokens'] = args['max_tokens']

def tune_parameters(provider, index=0,region='us-east-1'):
    st.subheader("Parameters")

    with st.form(key ='Form1'):
        models = utils.get_models(provider,region=region)
        model = st.selectbox('model', models,index=index)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = st.session_state['temperature'], step = 0.1)
        top_p = st.slider('top_p',min_value = 0.0, max_value = 1.0, value = st.session_state['top_p'], step = 0.1)
        max_tokens = st.number_input('max_tokens',min_value = 50, max_value = 4096, value = st.session_state['max_tokens'], step = 1)
        params = {
            "model":st.session_state['model'] , 
            "temperature":st.session_state['temperature'], 
            "top_p":st.session_state['top_p'],
            "max_tokens":st.session_state['max_tokens']
            }
        st.form_submit_button(label = 'Tune Parameters', on_click=update_parameters, kwargs=(params)) 


def invoke_model(client, prompt, model, accept = 'application/json', content_type = 'application/json',max_tokens  = 512, temperature = 1.0, top_p = 1.0):

    input = {
        'inputText': prompt,
        'textGenerationConfig': {
                'maxTokenCount': max_tokens,
                'stopSequences': stop_sequences,
                'temperature': temperature,
                'topP': top_p
        }
    }
    body=json.dumps(input)
    response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
    response_body = json.loads(response.get('body').read())
    results = response_body['results']
    for result in results:
        output = output + result['outputText']

    return output


def render_titan_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], 
        max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], 
        top_p=st.session_state['top_p'],
        model = st.session_state['model']
        )
    return output

