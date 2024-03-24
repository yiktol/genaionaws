import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as u_bedrock
import utils.stlib as stlib


params = {
	"model": "mistral.mistral-7b-instruct-v0:2",
	'max_tokens': 1024,
	'temperature': 0.1,
	'top_p': 0.9
}


def render_mistral_code(templatePath, suffix):
	env = Environment(loader=FileSystemLoader('templates'))
	template = env.get_template(templatePath)
	output = template.render(
		prompt=st.session_state[suffix]['prompt'], 
		max_tokens=st.session_state[suffix]['max_tokens'], 
		temperature=st.session_state[suffix]['temperature'], 
		top_p=st.session_state[suffix]['top_p'],
		model = st.session_state[suffix]['model']
		)
	return output


def mistral_generic(input_prompt, system_prompt=None):
	
	if system_prompt is None:
		system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
Please ensure that your responses are socially unbiased and positive in nature. \
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
If you don't know the answer to a question, please don't share false information."""
	else:
		system_prompt = system_prompt
		
	prompt = f"""<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n\n{input_prompt} [/INST]"""
	
	return prompt

def update_parameters(suffix,**args):
	for key in args:
		st.session_state[suffix][key] = args[key]
	return st.session_state[suffix]

def load_jsonl(file_path):
	d = []
	with jsonlines.open(file_path) as reader:
		for obj in reader:
			d.append(obj)
	return d

def tune_parameters(provider, suffix,index=0,region='us-east-1'):
	st.subheader("Parameters")

	with st.container(border=True):
		models = u_bedrock.getmodelIds('Mistral')
		model = st.selectbox(
			'model', models, index=models.index(u_bedrock.getmodelId(provider)))
		temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
		top_p = st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
		max_tokens = st.number_input('max_tokens',min_value = 50, max_value = 4096, value = 1024, step = 1)
		params = {
			"model":model, 
			"temperature":temperature, 
			"top_p":top_p,
			"max_tokens":max_tokens
			}
		col1, col2, col3 = st.columns([0.4,0.3,0.3])
		with col1:
			st.button(label = 'Tune Parameters', on_click=update_parameters, args=(suffix,), kwargs=(params))
		with col2:
			stlib.reset_session() 


def invoke_model(client, prompt, model, 
				 accept = 'application/json', 
				 content_type = 'application/json',
				 max_tokens = 512, 
				 temperature = 0.1, 
				 top_p = 0.9):
	output = ''
	input = {
		'prompt': mistral_generic(prompt),
		'max_tokens': max_tokens,
		'temperature': temperature,
		'top_p': top_p
	}
	body=json.dumps(input)
	response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read().decode('utf-8'))
	output = response_body.get('outputs')[0].get('text')

	return output



