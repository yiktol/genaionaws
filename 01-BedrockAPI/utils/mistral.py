import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as bedrock
import boto3


bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

params = {
	"model": "mistral.mistral-7b-instruct-v0:2",
	'max_tokens': 1024,
	'temperature': 0.1,
	'top_p': 0.9
}

accept = 'application/json'
content_type = 'application/json'

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
	
	prompt = f"""<s>[INST] {system_prompt} [/INST]\n<s>[INST] {input_prompt} [/INST]"""
	
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


def getmodelIds(providername='Mistral'):
	models =[]
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName']:
			models.append(model['modelId'])
			
	return models

def modelId():
	models = getmodelIds()
	model = st.selectbox(
		'model', models, index=models.index("mistral.mistral-7b-instruct-v0:2"))  
 
	return model

def tune_parameters():
	temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
	top_p = st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
	top_k = st.slider('top_k', min_value = 0, max_value = 200, value = 50, step = 1)
	max_tokens = st.number_input('max_tokens',min_value = 50, max_value = 4096, value = 512, step = 1)
	stop = st.text_input('stop_sequences', value = "User")
	params = {
		"temperature":temperature, 
		"top_p":top_p,
		"top_k":top_k,
		"max_tokens":max_tokens,
		"stop":[stop]
		}

	return params
 
 
def prompt_box(key, model, context=None, height=100, streaming=False,**params):
	response = ''
	with st.container(border=True):
		prompt = st.text_area("Enter your prompt here", value=context,
							  height=height,
							  key=f"Q{key}")
		submit = st.button("Submit", type="primary", key=f"S{key}")

	if submit:
		with st.spinner("Generating..."):
			if streaming:
				response = invoke_model_streaming(prompt, model, **params)
			else:
				response = invoke_model(prompt, model, **params)

	return response 
 
def invoke_model(prompt, model, **params):
	input = {
		'prompt': mistral_generic(prompt),
	}
	input.update(params)
	body=json.dumps(input)
	response = bedrock_runtime.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read().decode('utf-8'))
	outputs = response_body.get('outputs')

	return outputs


def invoke_model_streaming(prompt, model, **params):
	input = {
		'prompt': mistral_generic(prompt),
	}
	input.update(params)
	body=json.dumps(input)
	response = bedrock_runtime.invoke_model_with_response_stream(body=body, modelId=model, accept=accept,contentType=content_type)
	# response_body = json.loads(response.get('body').read().decode('utf-8'))
	# outputs = response_body.get('outputs')
 
	placeholder = st.empty()
	full_response = ''
	for event in response['body']:
		data = json.loads(event['chunk']['bytes'])
		chunk = data["outputs"][0]["text"]
		full_response += chunk
		placeholder.info(full_response)
	placeholder.info(full_response)

	# return outputs
