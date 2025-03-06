import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as bedrock
import boto3


bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

params = {
	"model": "meta.llama3-70b-instruct-v1:0",
	'max_gen_len': 1024,
	'temperature': 0.1,
	'top_p': 0.9
}

accept = 'application/json'
content_type = 'application/json'

def render_meta_code(templatePath, suffix):
	env = Environment(loader=FileSystemLoader('templates'))
	template = env.get_template(templatePath)
	output = template.render(
		prompt=st.session_state[suffix]['prompt'], 
		max_gen_len=st.session_state[suffix]['max_gen_len'], 
		temperature=st.session_state[suffix]['temperature'], 
		top_p=st.session_state[suffix]['top_p'],
		model = st.session_state[suffix]['model']
		)
	return output


def llama2_generic(input_prompt, system_prompt=None):
	
	if system_prompt is None:
		prompt = f"""<s>[INST] {input_prompt} [/INST]"""
# 		system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
# Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
# Please ensure that your responses are socially unbiased and positive in nature. \
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
# If you don't know the answer to a question, please don't share false information."""
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


def getmodelIds(providername='Meta'):
	models =[]
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if model['modelId'] in ['meta.llama2-13b-chat-v1:0:4k','meta.llama2-70b-chat-v1:0:4k','meta.llama2-13b-v1:0:4k','meta.llama2-70b-v1:0:4k']:
			continue
		if providername in model['providerName']:
			models.append(model['modelId'])
			
	return models

def modelId():
	models = getmodelIds()
	model = st.selectbox(
		'model', models, index=models.index("meta.llama3-70b-instruct-v1:0"))  
 
	return model


def tune_parameters():

	temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
	top_p = st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
	max_gen_len = st.number_input('max_gen_len',min_value = 50, max_value = 4096, value = 1024, step = 1)
	params = {
		"temperature":temperature, 
		"top_p":top_p,
		"max_gen_len":max_gen_len
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
	output = ''
	input = {
		'prompt': llama2_generic(prompt),
	}
	input.update(params)
	body=json.dumps(input)
	response = bedrock_runtime.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read())
	output = response_body['generation']

	return response_body



def invoke_model_streaming(prompt, model, **params):
	output = ''
	input = {
		'prompt': llama2_generic(prompt),
	}
	input.update(params)
	body=json.dumps(input)
	response = bedrock_runtime.invoke_model_with_response_stream(body=body, modelId=model, accept=accept,contentType=content_type)
	# response_body = json.loads(response.get('body').read())
	# output = response_body['generation']

	placeholder = st.empty()
	full_response = ''
	for event in response['body']:
		data = json.loads(event['chunk']['bytes'])
		chunk = data['generation']
		full_response += chunk
		placeholder.info(full_response)
	placeholder.info(full_response)


	# return response_body
