import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import boto3
import uuid

bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

params = {
	"model": "ai21.j2-ultra-v1",
	"maxTokens": 1024,
	"temperature": 0.1,
	"topP": 0.9,
	"stopSequences": [],
	"countPenalty": {
		"scale": 0
	},
	"presencePenalty": {
		"scale": 0
	},
	"frequencyPenalty": {
		"scale": 0
	}
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
		"Amazon" : "amazon.titan-tg1-large",
		"Anthropic" : "anthropic.claude-v2:1",
		"AI21" : "ai21.j2-ultra-v1",
		'Cohere': "cohere.command-text-v14",
		'Meta': "meta.llama2-70b-chat-v1",
		"Mistral": "mistral.mixtral-8x7b-instruct-v0:1",
		"Stability AI": "stability.stable-diffusion-xl-v1"
	}
	
	return model_mapping[providername]

def getmodelIds(providername='AI21'):
	models =[]
	bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName']:
			models.append(model['modelId'])
			
	return models

def modelId():
	models = getmodelIds()
	model = st.selectbox(
		'model', models, index=models.index("ai21.j2-ultra-v1"))  
 
	return model

def render_jurassic_code(templatePath, suffix):
	env = Environment(loader=FileSystemLoader('templates'))
	template = env.get_template(templatePath)
	output = template.render(
		prompt=st.session_state[suffix]['prompt'], 
		maxTokens=st.session_state[suffix]['maxTokens'], 
		temperature=st.session_state[suffix]['temperature'], 
		topP=st.session_state[suffix]['topP'],
		model = st.session_state[suffix]['model']
		)
	return output


def update_parameters(suffix,**args):
	for key in args:
		st.session_state[suffix][key] = args[key]


def load_jsonl(file_path):
	d = []
	with jsonlines.open(file_path) as reader:
		for obj in reader:
			d.append(obj)
	return d

def tune_parameters():
	temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
	topP = st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
	maxTokens = st.number_input('maxTokens',min_value = 50, max_value = 4096, value = 1024, step = 1)
	stopSequences = st.text_input('stopSequences', value = 'User:')
	countPenalty = st.slider('countPenalty', min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1)
	presencePenalty = st.slider('presencePenalty', min_value = 0.0, max_value = 5.0, value = 0.0, step = 0.25)
	frequencyPenalty = st.slider('frequencyPenalty', min_value = 0.0, max_value = 500.0, value = 0.0, step = 10.0)

	params = {
		'maxTokens': maxTokens,
		'temperature': temperature,
		'topP': topP,
		'stopSequences': [stopSequences],
		'countPenalty': {'scale': countPenalty},
		'presencePenalty': {'scale': presencePenalty},
		'frequencyPenalty': {'scale': frequencyPenalty}
		}
  
	  
	return params

def prompt_box(key, model, context=None, height=100, **params):
	response = ''
	with st.container(border=True):
		prompt = st.text_area("Enter your prompt here", value=context,
							  height=height,
							  key=f"Q{key}")
		submit = st.button("Submit", type="primary", key=f"S{key}")

	if submit:
		with st.spinner("Generating..."):
			response = invoke_model(
				bedrock_runtime,
				prompt,
				model=model,
				**params)

	return response


def invoke_model(client, prompt, model, 
				 accept = 'application/json', 
				 content_type = 'application/json',
				 **params):
	output = ''
	input = {
		'prompt': prompt, 
	}
	  
	input.update(params)
	body=json.dumps(input)
	response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read())
	completions = response_body['completions']
	for part in completions:
		output = output + part['data']['text']

	return output



