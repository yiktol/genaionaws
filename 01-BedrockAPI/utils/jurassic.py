import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as u_bedrock
import utils.stlib as stlib


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

def tune_parameters(provider, suffix,index=0,region='us-east-1'):
	st.subheader("Parameters")

	with st.container(border=True):
		models = u_bedrock.getmodelIds('AI21')
		model = st.selectbox(
			'model', models, index=models.index(u_bedrock.getmodelId(provider)))
		temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
		topP = st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
		maxTokens = st.number_input('maxTokens',min_value = 50, max_value = 4096, value = 1024, step = 1)
		params = {
			"model":model, 
			"temperature":temperature, 
			"topP":topP,
			"maxTokens":maxTokens
			}
		col1, col2, col3 = st.columns([0.4,0.3,0.3])
		with col1:
			st.button(label = 'Tune Parameters', on_click=update_parameters, args=(suffix,), kwargs=(params))
		with col2:
			stlib.reset_session() 


def invoke_model(client, prompt, model, 
				 accept = 'application/json', 
				 content_type = 'application/json',
				 maxTokens = 512, 
				 temperature = 0.1, 
				 topP = 0.9,
				 stop_sequences = []):
	output = ''
	input = {
		'prompt': prompt, 
		'maxTokens': maxTokens,
		'temperature': temperature,
		'topP': topP,
		'stopSequences': stop_sequences,
		'countPenalty': {'scale': 0},
		'presencePenalty': {'scale': 0},
		'frequencyPenalty': {'scale': 0}
	}
	body=json.dumps(input)
	response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read())
	completions = response_body['completions']
	for part in completions:
		output = output + part['data']['text']

	return output



