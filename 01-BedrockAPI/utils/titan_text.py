import streamlit as st
import jsonlines
import json
import boto3
from jinja2 import Environment, FileSystemLoader


bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

accept = 'application/json'
content_type = 'application/json'

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


def getmodelIds():
	models = []
	available_models = bedrock.list_foundation_models()

	for model in available_models['modelSummaries']:
		if "amazon.titan-tg1-large" in model['modelId'] or "amazon.titan-text" in model['modelId']:
			models.append(model['modelId'])

	return models



def titan_generic(input_prompt):
	prompt = f"""User: {input_prompt}\n\nBot:"""
	return prompt


def load_jsonl(file_path):
	d = []
	with jsonlines.open(file_path) as reader:
		for obj in reader:
			d.append(obj)
	return d


def modelId():
	models = getmodelIds()
	model = st.selectbox(
		'model', models, index=models.index("amazon.titan-tg1-large"))  
 
	return model

def tune_parameters():
	temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
	topP = st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
	maxTokenCount = st.number_input('maxTokenCount',min_value = 50, max_value = 4096, value = 1024, step = 1)
	stopSequences = st.text_input('stopSequences', value = "User:")
	params = {
		"temperature":temperature, 
		"topP":topP,
		"maxTokenCount":maxTokenCount,
		"stopSequences":[stopSequences]
		}

	return params


def prompt_box(key, model, context=None, height=100, streaming=False,**params):
	response = ''
	with st.container(border=True):
		prompt = st.text_area("Enter your prompt here", value=context,
							  height=height,key=f"Q{key}")
		submit = st.button("Submit", type="primary", key=f"S{key}")

	if submit:
		if context is not None:
			prompt = context + "\n\n" + prompt
			
		prompt = titan_generic(prompt)
		with st.spinner("Generating..."):
			if streaming:
				response = invoke_model_streaming(prompt,model,**params)
			else:
				response = invoke_model(prompt,model,**params)
			
	return response

def invoke_model(prompt, model, **params):
	output = ''
	input = {
		'inputText': titan_generic(prompt),
		'textGenerationConfig': params
	}
	body=json.dumps(input)
	response = bedrock_runtime.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
	response_body = json.loads(response.get('body').read())
	results = response_body['results']
	for result in results:
		output = output + result['outputText']

	return output


def invoke_model_streaming(prompt, model,**params):
	
	input = {
		'inputText': titan_generic(prompt),
		'textGenerationConfig': params
	}
	body=json.dumps(input)
	response = bedrock_runtime.invoke_model_with_response_stream(
				body=body, 
				modelId=model, 
				accept=accept, 
				contentType=content_type
			)

	placeholder = st.empty()
	full_response = ''
	for event in response['body']:
		data = json.loads(event['chunk']['bytes'])
		chuck = data['outputText']
		full_response += chuck
		placeholder.info(full_response)
	placeholder.info(full_response)
		
	return response

