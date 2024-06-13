import streamlit as st
import jsonlines
import json
import boto3
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as u_bedrock
import utils.stlib as stlib

bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime', region_name='us-east-1')


def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d


params = {
    "temperature": 0.1,
    "p": 0.9,
    "k": 50,
    "max_tokens": 2048,
    "stop_sequences": "'.'",
    "stream": False,
    "num_generations": 1,
    "return_likelihoods": "NONE",
    "model": "cohere.command-text-v14",
}

accept = 'application/json'
content_type = 'application/json'


def getmodelIds():
    models = []
    available_models = bedrock.list_foundation_models()

    for model in available_models['modelSummaries']:
        if model['modelId'] in ['cohere.command-text-v14:7:4k','cohere.command-light-text-v14:7:4k']:
            continue
        elif "cohere.command" in model['modelId']:
            models.append(model['modelId'])

    return models


def modelId():
    models = getmodelIds()
    model = st.selectbox(
        'model', models, index=models.index("cohere.command-text-v14"))

    return model


def render_cohere_code(templatePath, suffix):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state[suffix]['prompt'],
        max_tokens=st.session_state[suffix]['max_tokens'],
        temperature=st.session_state[suffix]['temperature'],
        p=st.session_state[suffix]['p'],
        k=st.session_state[suffix]['k'],
        model=st.session_state[suffix]['model'],
        stop_sequences=st.session_state[suffix]['stop_sequences'],
        return_likelihoods=st.session_state[suffix]['return_likelihoods']
    )
    return output


def cohere_generic(input_prompt):
    prompt = input_prompt
    return prompt


def update_parameters(suffix, **args):
    for key in args:
        st.session_state[suffix][key] = args[key]
    return st.session_state[suffix]


def prompt_box(key, model, context=None, height=100, streaming=False, **params):
    response = ''
    with st.container(border=True):
        prompt = st.text_area("Enter your prompt here", value=context,
                              height=height,
                              key=f"Q{key}")
        submit = st.button("Submit", type="primary", key=f"S{key}")

    if submit:
        with st.spinner("Generating..."):
            if streaming:
                params.update({"stream": True})
                response = invoke_model_streaming(prompt, model, **params)
            else:
                response = invoke_model(prompt, model, **params)

    return response


def tune_parameters(model):
    params = ''
    temperature = st.slider(
        'temperature', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    p = st.slider('p', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    k = st.slider('k', min_value=0, max_value=100, value=50, step=1)
    max_tokens = st.number_input(
        'max_tokens', min_value=50, max_value=4096, value=2048, step=1)
    stop_sequences = st.text_input('stop_sequences', value='User:')
    # stream = st.checkbox('stream', value=True, disabled=True)
    if not model.startswith('cohere.command-r'):
        num_generations = st.slider(
            'num_generations', min_value=1, max_value=5, value=1, step=1)
        return_likelihoods = st.radio(
            'return_likelihoods', ["NONE", "ALL", "GENERATION"], horizontal=True)
        params = {
            "temperature": temperature,
            "p": p,
            "k": k,
            "stop_sequences": [stop_sequences],
            "max_tokens": max_tokens,
            # "stream": stream,
            "num_generations": num_generations,
            "return_likelihoods": return_likelihoods
        }
    elif model.startswith('cohere.command-r'):
        params = {
            "temperature": temperature,
            "p": p,
            "k": k,
            "stop_sequences": [stop_sequences],
            "max_tokens": max_tokens,
            # "stream": stream,
        }

    return params


def invoke_model(prompt, model, **params):

    output = ''
    input =''
    if not model.startswith('cohere.command-r'):
        input = {
            'prompt': prompt
        }
    elif model.startswith('cohere.command-r'):
        input = {
            'message': prompt
        }

    input.update(params)
    body = json.dumps(input)
    response = bedrock_runtime.invoke_model(
        body=body, modelId=model, accept=accept, contentType=content_type)
    response_body = json.loads(response.get('body').read())
    results = response_body['generations']
    # results = response_body['text']
    # for result in results:
    # 	output = output + result['text']

    return results


def invoke_model_streaming(prompt, model, **params):

    output = ''
    input = {
        'prompt': prompt
    }
    input.update(params)
    body = json.dumps(input)
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, modelId=model, accept=accept, contentType=content_type)
    # response_body = json.loads(response['body'])
    # results = response_body['generations']

    # print(response_body)

    placeholder = st.empty()
    full_response = ''
    for event in response['body']:
        # print(event)
        data = json.loads(event['chunk']['bytes'])
        if not data["is_finished"]:
            chunk = data["text"]
        full_response += chunk
        placeholder.info(full_response)
    placeholder.info(full_response)

# return response
