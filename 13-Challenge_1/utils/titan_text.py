import streamlit as st
import jsonlines
import json
import boto3
import uuid
from jinja2 import Environment, FileSystemLoader



params = {
	"model": "amazon.titan-tg1-large",
	"temperature":0.1, 
	"topP":0.9,
	"maxTokenCount":1024,
	"stopSequences": []
	}


def initsessionkeys(params, suffix):
    if suffix not in st.session_state:
        st.session_state[suffix] = {}
    for key in params.keys():
        if key not in st.session_state[suffix]:
            st.session_state[suffix][key] = params[key]
    return st.session_state[suffix]

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

def getmodelIds(providername):
    models =[]
    bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
    available_models = bedrock.list_foundation_models()
    
    for model in available_models['modelSummaries']:
        if providername in model['providerName']:
            models.append(model['modelId'])
            
    return models

def tune_parameters(provider='Amazon'):

	# models = getmodelIds('Amazon')
	# model = st.selectbox(
	# 	'model', models, index=models.index(getmodelId(provider)))
	temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
	topP = st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
	maxTokenCount = st.number_input('maxTokenCount',min_value = 50, max_value = 4096, value = 1024, step = 1)
	params = {
		# "model":model, 
		"temperature":temperature, 
		"topP":topP,
		"maxTokenCount":maxTokenCount
		}

	return params


def invoke_model(client, prompt, model, 
				 accept = 'application/json', 
				 content_type = 'application/json',
				 **params):
	output = ''
	input = {
		'inputText': titan_generic(prompt),
		'textGenerationConfig': params
	}
	body=json.dumps(input)
	response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read())
	results = response_body['results']
	for result in results:
		output = output + result['outputText']

	return output



