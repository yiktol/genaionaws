import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as u_bedrock
import utils.stlib as stlib


params = {
	"model": "amazon.titan-tg1-large",
	"temperature":0.1, 
	"topP":0.9,
	"maxTokenCount":1024,
	"stopSequences": []
	}


def render_titan_code(templatePath, suffix):
	env = Environment(loader=FileSystemLoader('templates'))
	template = env.get_template(templatePath)
	output = template.render(
		prompt=st.session_state[suffix]['prompt'], 
		maxTokenCount=st.session_state[suffix]['maxTokenCount'], 
		temperature=st.session_state[suffix]['temperature'], 
		topP=st.session_state[suffix]['topP'],
		stopSequences=st.session_state[suffix]['stopSequences'],
		model = st.session_state[suffix]['model']
		)
	return output


def titan_generic(input_prompt):
	prompt = f"""User: {input_prompt}\n\nAssistant:"""
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
		models = u_bedrock.getmodelIds('Amazon')
		model = st.selectbox(
			'model', models, index=models.index(u_bedrock.getmodelId(provider)))
		temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = st.session_state[suffix]['temperature'], step = 0.1)
		topP = st.slider('topP',min_value = 0.0, max_value = 1.0, value = st.session_state[suffix]['topP'], step = 0.1)
		maxTokenCount = st.number_input('maxTokenCount',min_value = 50, max_value = 4096, value = st.session_state[suffix]['maxTokenCount'], step = 1)
		params = {
			"model":model, 
			"temperature":temperature, 
			"topP":topP,
			"maxTokenCount":maxTokenCount
			}
		col1, col2, col3 = st.columns([0.4,0.3,0.3])
		with col1:
			st.button(label = 'Tune Parameters', on_click=update_parameters, args=(suffix,), kwargs=(params))
		with col2:
			stlib.reset_session() 


def invoke_model(client, prompt, model, 
				 accept = 'application/json', 
				 content_type = 'application/json',
				 maxTokenCount = 512, 
				 temperature = 0.1, 
				 topP = 0.9,
				 stop_sequences = []):
	output = ''
	input = {
		'inputText': titan_generic(prompt),
		'textGenerationConfig': {
				'maxTokenCount': maxTokenCount,
				'stopSequences': stop_sequences,
				'temperature': temperature,
				'topP': topP
		}
	}
	body=json.dumps(input)
	response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read())
	results = response_body['results']
	for result in results:
		output = output + result['outputText']

	return output



