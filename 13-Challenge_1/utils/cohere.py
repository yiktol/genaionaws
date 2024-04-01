import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import boto3
import uuid


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
	"max_tokens": 200,
	"stop_sequences": '""',
	"return_likelihoods": "NONE",
	"model": "cohere.command-text-v14",
}


def initsessionkeys(params, suffix):
    if suffix not in st.session_state:
        st.session_state[suffix] = {}
    for key in params.keys():
        if key not in st.session_state[suffix]:
            st.session_state[suffix][key] = params[key]
    return st.session_state[suffix]


def reset_session():
    def form_callback():
        for key in st.session_state.keys():
            del st.session_state[key]

    st.button(label='Reset', on_click=form_callback, key=uuid.uuid1())


def getmodelId(providername):
    model_mapping = {
        "Amazon": "amazon.titan-tg1-large",
        "Anthropic": "anthropic.claude-v2:1",
        "AI21": "ai21.j2-ultra-v1",
        'Cohere': "cohere.command-text-v14",
        'Meta': "meta.llama2-70b-chat-v1",
        "Mistral": "mistral.mixtral-8x7b-instruct-v0:1",
        "Stability AI": "stability.stable-diffusion-xl-v1"
    }

    return model_mapping[providername]


def getmodelIds(providername):
    models = []
    bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
    available_models = bedrock.list_foundation_models()

    for model in available_models['modelSummaries']:
        if providername in model['providerName']:
            models.append(model['modelId'])

    return models


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


def tune_parameters():
	temperature = st.slider('temperature', min_value=0.0,
	                        max_value=1.0, value=0.1, step=0.1)
	p = st.slider('p', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
	k = st.slider('k', min_value=0, max_value=100, value=50, step=1)
	max_tokens = st.number_input(
		'max_tokens', min_value=50, max_value=2048, value=200, step=1)
	stop_sequences = st.text_input('stop_sequences', value='""')
	return_likelihoods = st.text_input('return_likelihoods', value="NONE")
	params = {
            "temperature": temperature,
          		"p": p,
          		"k": k,
          		"stop_sequences": [stop_sequences],
          		"max_tokens": max_tokens,
          		"return_likelihoods": return_likelihoods
        }

	return params


def invoke_model(client, prompt, model,
                 accept='application/json',
                 content_type='application/json',
                 **params):
	output = ''
	input = {
		'prompt': prompt
	}

	input.update(params)
	body = json.dumps(input)
	response = client.invoke_model(
		body=body, modelId=model, accept=accept, contentType=content_type)
	response_body = json.loads(response.get('body').read())
	results = response_body['generations']
	for result in results:
		output = output + result['text']

	return output
