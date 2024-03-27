import streamlit as st
import jsonlines
import json
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as bedrock
import utils.stlib as stlib


params = {
	"model": "ai21.j2-ultra-v1",
	"maxTokens": 200,
	"temperature": 0.5,
	"topP": 0.5,
	"stopSequences": [],
	"countPenalty_scale": 0,
	"presencePenalty_scale": 0,
	"frequencyPenalty_scale": 0
}


def render_jurassic_code(templatePath, suffix):
	env = Environment(loader=FileSystemLoader('templates'))
	template = env.get_template(templatePath)
	output = template.render(
		prompt=st.session_state[suffix]['prompt'], 
		maxTokens=st.session_state[suffix]['maxTokens'], 
		temperature=st.session_state[suffix]['temperature'], 
		topP=st.session_state[suffix]['topP'],
		model = st.session_state[suffix]['model'],
        stopSequences = st.session_state[suffix]['stopSequences'],
		countPenalty = st.session_state[suffix]['countPenalty_scale'],
		presencePenalty = st.session_state[suffix]['presencePenalty_scale'],
		frequencyPenalty = st.session_state[suffix]['frequencyPenalty_scale']
		)
	return output


def load_jsonl(file_path):
	d = []
	with jsonlines.open(file_path) as reader:
		for obj in reader:
			d.append(obj)
	return d

def tune_parameters(provider, suffix,index=0,region='us-east-1'):
	st.subheader("Parameters")

	with st.form("jurassic-form"):
		models = bedrock.getmodelIds('AI21')
		model = st.selectbox(
			'model', models, index=models.index(bedrock.getmodelId(provider)))
		temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
		topP = st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
		maxTokens = st.number_input('maxTokens',min_value = 50, max_value = 8191, value = 200, step = 100)
		stopSequences = st.text_input('stopSequences', value = ['.'])
		countPenalty = st.slider('countPenalty', min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1)
		presencePenalty = st.slider('presencePenalty', min_value = 0.0, max_value = 5.0, value = 0.0, step = 0.25)
		frequencyPenalty = st.slider('frequencyPenalty', min_value = 0.0, max_value = 500.0, value = 0.0, step = 10.0)

		params = {
			"model":model, 
			"temperature":temperature, 
			"topP":topP,
			"maxTokens":maxTokens,
			"stopSequences":stopSequences,
			"countPenalty_scale":countPenalty,
			"presencePenalty_scale":presencePenalty,
			"frequencyPenalty_scale":frequencyPenalty
			}
  
		st.form_submit_button(label = 'Tune Parameters', on_click=stlib.update_parameters, args=(suffix,), kwargs=(params))



def invoke_model(client, prompt, model, 
				 accept = 'application/json', 
				 content_type = 'application/json',
				 maxTokens = 512, 
				 temperature = 0.1, 
				 topP = 0.9,
				 stopSequences = [],
     			 countPenalty = 0,
				 presencePenalty = 0,
				frequencyPenalty = 0):
	output = ''
	input = {
		'prompt': prompt, 
		'maxTokens': maxTokens,
		'temperature': temperature,
		'topP': topP,
		'stopSequences': stopSequences,
		'countPenalty': {'scale': countPenalty},
		'presencePenalty': {'scale': presencePenalty},
		'frequencyPenalty': {'scale': frequencyPenalty}
	}
	body=json.dumps(input)
	response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read())
	completions = response_body['completions']
	for part in completions:
		output = output + part['data']['text']

	return output



