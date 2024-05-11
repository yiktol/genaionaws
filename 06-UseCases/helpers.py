import streamlit as st
import json
import boto3
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm

def set_page_config():
	st.set_page_config( 
	page_title="Use Cases",  
	page_icon=":rock:",
	layout="wide",
	initial_sidebar_state="expanded",
)
	
def bedrock_runtime_client(region_name='us-east-1'):
	bedrock_runtime = boto3.client(
	service_name='bedrock-runtime',
	region_name=region_name, 
	)
	return bedrock_runtime

def bedrock_client(region_name='us-east-1'):
	bedrock = boto3.client(
		service_name='bedrock',
		region_name=region_name, 
		)
	return bedrock


def claude_generic(input_prompt):
	prompt = f"""Human: {input_prompt}\n\nAssistant:"""
	return prompt

def titan_generic(input_prompt):
	prompt = f"""User: {input_prompt}\n\nAssistant:"""
	return prompt

def llama2_generic(input_prompt, system_prompt=None):
	if system_prompt is None:
		prompt = f"<s>[INST] {input_prompt} [/INST]"
	else:
		prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{input_prompt} [/INST]"
	return prompt

def mistral_generic(input_prompt):
	prompt = f"<s>[INST] {input_prompt} [/INST]"
	return prompt

def get_embedding(bedrock, text):
	modelId = 'amazon.titan-embed-text-v1'
	accept = 'application/json'
	contentType = 'application/json'
	input = {
			'inputText': text
		}
	body=json.dumps(input)
	response = bedrock.invoke_model(
		body=body, modelId=modelId, accept=accept,contentType=contentType)
	response_body = json.loads(response.get('body').read())
	embedding = response_body['embedding']
	return embedding

def calculate_distance(v1, v2):
	distance = math.dist(v1, v2)
	return distance

def calculate_dot_product(v1, v2):
	similarity = dot(v1, v2)
	return similarity

def calculate_cosine_similarity(v1, v2):
	similarity = dot(v1, v2)/(norm(v1)*norm(v2))
	return similarity

def search(dataset, v):
	distance_array=[]
	for item in dataset:
		item['distance'] = calculate_distance(item['embedding'], v)
		distance_array.append(item['distance'])
	dataset.sort(key=lambda x: x['distance'])
	return [dataset[0]['text'],distance_array]



def classify(classes, v):
	distance_array=[]
	for item in classes:
		item['distance'] = calculate_distance(item['embedding'], v)
		distance_array.append(item['distance'])
	classes.sort(key=lambda x: x['distance'])
	return [classes[0]['name'],distance_array]
  
def find_outliers_by_count(dataset, count):
	# find the center of mass
	embeddings = []
	for item in dataset:
		embeddings.append(item['embedding'])
	center = np.mean(embeddings, axis=0)
	# calculate distance from center
	distances=[]
	for item in dataset:
		item['distance'] = calculate_distance(item['embedding'], center)
		distances.append(item['distance'])
	sd = np.std(embeddings)
	# sort the distances in reverse order
	dataset.sort(key=lambda x: x['distance'], reverse=True)
	# return N outliers
	return [dataset[0:count],distances]

def find_outliers_by_percentage(dataset, percent):
	# find the center of mass
	embeddings = []
	for item in dataset:
		embeddings.append(item['embedding'])
	center = np.mean(embeddings, axis=0)
	# calculate distance from center
	for item in dataset:
		item['distance'] = calculate_distance(item['embedding'], center)
	# sort the distances in reverse order
	dataset.sort(key=lambda x: x['distance'], reverse=True)
	# return top x% outliers
	total = len(dataset)
	count = math.floor(percent * total / 100)
	return dataset[0:count]  

def find_outliers_by_distance(dataset, percent):
	# find the center of mass
	embeddings = []
	for item in dataset:
		embeddings.append(item['embedding'])
	center = np.mean(embeddings, axis=0)
	# calculate distance from center
	for item in dataset:
		item['distance'] = calculate_distance(item['embedding'], center)
	# sort the distances in reverse order
	dataset.sort(key=lambda x: x['distance'], reverse=True)
	# return outliers beyond x% of max distance
	max_distance = dataset[0]['distance']
	min_distance = percent * max_distance / 100
	outliers = []
	for item in dataset:
		if item['distance'] >= min_distance:
			outliers.append(item)
	return outliers  
  

def getmodelId(providername):
	model_mapping = {
		"Amazon" : "amazon.titan-tg1-large",
		"Anthropic" : "anthropic.claude-v2:1",
		"AI21" : "ai21.j2-ultra-v1",
		'Cohere': "cohere.command-text-v14",
		'Meta': "meta.llama2-70b-chat-v1",
		"Mistral": "mistral.mixtral-8x7b-instruct-v0:1"
	}
	
	return model_mapping[providername]

def getmodel_index(providername):
	
	default_model = getmodelId(providername)
	
	idx = getmodelIds(providername).index(default_model)
	
	return idx

def getmodelIds(providername):
	models =[]
	bedrock =bedrock_client()
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName']:
			models.append(model['modelId'])
			
	return models


def getmodelparams(providername):
	model_mapping = {
		"Amazon" : {
			"maxTokenCount": 4096,
			"stopSequences": [], 
			"temperature": 0.1,
			"topP": 0.9
			},
		"Anthropic" : {
			"max_tokens_to_sample": 4096,
			"temperature": 0.1,
			"top_k": 250,
			"top_p": 0.9,
			"stop_sequences": ["\n\nHuman"],
			},
		"AI21" : {
			"maxTokens": 4096,
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
		},
		"Cohere": {
			"max_tokens": 4096,
			"temperature": 0.1,
			"p": 0.9,
			"k": 0,
			"stop_sequences": [],
			"return_likelihoods": "NONE"
		},
		"Meta":{ 
			'max_gen_len': 1024,
			'top_p': 0.9,
			'temperature': 0.1
		}
	}
	
	return model_mapping[providername]


def invoke_model(client, prompt, model, 
	accept = 'application/json', content_type = 'application/json',
	max_tokens  = 512, temperature = 1.0, top_p = 1.0, top_k = 200, stop_sequences = [],
	count_penalty = 0, presence_penalty = 0, frequency_penalty = 0, return_likelihoods = 'NONE'):
	# default response
	output = ''
	# identify the model provider
	provider = model.split('.')[0] 
	# InvokeModel
	if ('anthropic.claude-3' in model): 
		input = {
			'max_tokens': max_tokens, 
			'temperature': temperature,
			'top_k': top_k,
			'top_p': top_p,
			'stop_sequences': stop_sequences,
			"anthropic_version": "bedrock-2023-05-31",
			"messages": [{"role": "user", "content": prompt}]
		}

		# input.update(params)
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept, contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body.get('content')[0]['text']
	elif (provider == 'anthropic'): 
		input = {
			'prompt': claude_generic(prompt),
			'max_tokens_to_sample': max_tokens, 
			'temperature': temperature,
			'top_k': top_k,
			'top_p': top_p,
			'stop_sequences': stop_sequences
		}
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body['completion']
	elif (provider == 'ai21'): 
		input = {
			'prompt': prompt, 
			'maxTokens': max_tokens,
			'temperature': temperature,
			'topP': top_p,
			'stopSequences': stop_sequences,
			'countPenalty': {'scale': count_penalty},
			'presencePenalty': {'scale': presence_penalty},
			'frequencyPenalty': {'scale': frequency_penalty}
		}
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		completions = response_body['completions']
		for part in completions:
			output = output + part['data']['text']
	elif (provider == 'amazon'): 
		input = {
			'inputText': titan_generic(prompt),
			'textGenerationConfig': {
				  'maxTokenCount': max_tokens,
				  'stopSequences': stop_sequences,
				  'temperature': temperature,
				  'topP': top_p
			}
		}
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		results = response_body['results']
		for result in results:
			output = output + result['outputText']
	elif (provider == 'cohere'): 
		input = {
			'prompt': prompt, 
			'max_tokens': max_tokens,
			'temperature': temperature,
			'k': top_k,
			'p': top_p,
			'stop_sequences': stop_sequences,
			'return_likelihoods': return_likelihoods
		}
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		results = response_body['generations']
		for result in results:
			output = output + result['text']
	elif (provider == 'meta'): 
		input = {
			'prompt': llama2_generic(prompt),
			'max_gen_len': max_tokens,
			'temperature': temperature,
			'top_p': top_p
		}
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body['generation']
	elif (provider == 'mistral'): 
		input = {
			'prompt': mistral_generic(prompt),
			'max_tokens': max_tokens,
			'temperature': temperature,
			'top_p': top_p,
			'top_k': top_k
		}
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body.get('outputs')[0].get('text')
	return output


