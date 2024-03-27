import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as u_bedrock
import utils.stlib as stlib


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


def tune_parameters(provider, suffix, index=0, region='us-east-1'):
	st.subheader("Parameters")

	with st.form("cohere-form"):
		models = u_bedrock.getmodelIds('Cohere')
		model = st.selectbox(
			'model', models, index=models.index(u_bedrock.getmodelId(provider)))
		temperature = st.slider(
			'temperature', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
		p = st.slider('p', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
		k = st.slider('k', min_value=0, max_value=100, value=50, step=1)
		max_tokens = st.number_input(
			'max_tokens', min_value=50, max_value=4096, value=2048, step=1)
		stop_sequences = st.text_input('stop_sequences', value="'.'")
		stream = st.checkbox('stream', value=False, disabled=True)
		num_generations = st.slider(
			'num_generations', min_value=1, max_value=5, value=1, step=1)
		return_likelihoods = st.radio(
			'return_likelihoods', ["NONE", "ALL", "GENERATION"], horizontal=True)
		params = {
			"model": model,
			"temperature": temperature,
			"p": p,
			"k": k,
			"stop_sequences": stop_sequences,
			"max_tokens": max_tokens,
			"stream": stream,
			"num_generations": num_generations,
			"return_likelihoods": return_likelihoods
		}

		st.form_submit_button(label='Tune Parameters', on_click=update_parameters, args=(
			suffix,), kwargs=(params))



def invoke_model(client, prompt, model,
				 accept='application/json',
				 content_type='application/json',
				 max_tokens=2048,
				 temperature=0.1,
				 p=0.9,
				 k=50,
				 stop_sequences=[],
				 stream=False,
				 num_generations=1,
				 return_likelihoods="NONE"):
	output = ''
	input = {
		'prompt': prompt,
		'max_tokens': max_tokens,
		'temperature': temperature,
		'k': k,
		'p': p,
		'stop_sequences': [stop_sequences],
		"stream": stream,
		"num_generations": num_generations,
		"return_likelihoods": return_likelihoods
	}
	body = json.dumps(input)
	response = client.invoke_model(
		body=body, modelId=model, accept=accept, contentType=content_type)
	response_body = json.loads(response.get('body').read())
	generations = response_body.get('generations')

	# for result in results:
	# 	output = output + result['text']

	return generations
