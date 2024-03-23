import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
from utils.bedrock import get_models


def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d

params = {  "model": "anthropic.claude-v2",
            "max_tokens_to_sample": 1024,
            "temperature": 0.1,
            "top_k": 50,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman"],
            }


def render_claude_code(templatePath,suffix):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state[suffix]['prompt'], 
        max_tokens_to_sample=st.session_state[suffix]['max_tokens_to_sample'], 
        temperature=st.session_state[suffix]['temperature'], 
        top_p = st.session_state[suffix]['top_p'],
        top_k = st.session_state[suffix]['top_k'],
        model = st.session_state[suffix]['model'],
        stop_sequences = st.session_state[suffix]['stop_sequences']
        )
    return output


def claude_generic(input_prompt):
    prompt = f"""Human: {input_prompt}\n\nAssistant:"""
    return prompt

def update_parameters(suffix,**args):
    for key in args:
        st.session_state[suffix][key] = args[key]
    return st.session_state[suffix]

def tune_parameters(provider, suffix,index=0,region='us-east-1'):
    st.subheader("Parameters")

    with st.container(border=True):
        models = get_models(provider,region=region)
        model = st.selectbox('model', models,index=index)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
        top_p = st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        top_k = st.slider('top_k', min_value = 0, max_value = 100, value = 50, step = 1)
        max_tokens_to_sample = st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 1024, step = 1)
        stop_sequences = st.text_input('stop_sequences', value = ["\n\nHuman"])
        params = {
            "model":model, 
            "temperature":temperature, 
            "top_p":top_p,
            "top_k":top_k,
            "stop_sequences":stop_sequences,
            "max_tokens_to_sample":max_tokens_to_sample
            }
        st.button(label = 'Tune Parameters', on_click=update_parameters, args=(suffix,), kwargs=(params)) 


def invoke_model(client, prompt, model, 
                 accept = 'application/json', 
                 content_type = 'application/json',
                 max_tokens_to_sample = 512, 
                 temperature = 0.1, 
                 top_p = 0.9,
                 top_k = 50,
                 stop_sequences = ["\n\nHuman"]):
    output = ''
    input = {
        'prompt': claude_generic(prompt),
        'max_tokens_to_sample': max_tokens_to_sample,
        'stop_sequences': stop_sequences,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        "anthropic_version": "bedrock-2023-05-31"
        }
    body=json.dumps(input)
    response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
    response_body = json.loads(response.get('body').read())
    output = response_body['completion']

    return output



