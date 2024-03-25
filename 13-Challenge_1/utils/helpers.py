import streamlit as st
import jsonlines
import json
import boto3
import utils.titan_text as titan
import utils.claude2 as claude2
import utils.llama2 as llama2
import utils.mistral as mistral
import utils.cohere as cohere
import utils.jurassic as jurassic


def tune_parameters(provider):
	match provider:
		case "Anthropic":
			claude2.initsessionkeys(claude2.params, 'claude2')
			claude2.tune_parameters()
		case "Amazon":
			titan.initsessionkeys(titan.params, 'titan')
			titan.tune_parameters()
		case "Cohere":
			cohere.initsessionkeys(cohere.params, 'cohere')
			cohere.tune_parameters()
		case "AI21":
			jurassic.initsessionkeys(llama2.params, 'jurassic')          
			jurassic.tune_parameters()
		case "Mistral":
			mistral.initsessionkeys(mistral.params, 'mistral')
			mistral.tune_parameters()
		case "Meta":
			llama2.initsessionkeys(llama2.params, 'llama2')
			llama2.tune_parameters()
		case _:
			print("Provider not supported")
			return False


def set_page_config():
	st.set_page_config( 
	page_title="Challenge",  
	page_icon=":rock:",
	layout="wide",
	initial_sidebar_state="expanded",
)
	
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
	   
	

def load_jsonl(file_path):
	d = []
	with jsonlines.open(file_path) as reader:
		for obj in reader:
			d.append(obj)
	return d


def client(region='us-east-1'):
  return boto3.client(
	service_name='bedrock',
	region_name=region
  )
  
  
def runtime_client(region='us-east-1'):
	bedrock_runtime = boto3.client(
	service_name='bedrock-runtime',
	region_name=region, 
	)
	return bedrock_runtime  


def reset_session():
	def form_callback():
		for key in st.session_state.keys():
			del st.session_state[key]


	st.button(label='Reset', on_click=form_callback)


def getmodelId(providername):
	model_mapping = {
		"Amazon" : "amazon.titan-tg1-large",
		"Anthropic" : "anthropic.claude-v2:1",
		"AI21" : "ai21.j2-ultra-v1",
		'Cohere': "cohere.command-text-v14",
		'Meta': "meta.llama2-70b-chat-v1",
		"Mistral": "mistral.mixtral-8x7b-instruct-v0:1",
		"Stability AI": "stability.stable-diffusion-xl-v1",
  		"Anthropic Claude 3" : "anthropic.claude-3-sonnet-20240229-v1:0"
	}
	
	return model_mapping[providername]

def getmodel_index(providername):
	
	default_model = getmodelId(providername)
	
	idx = getmodelIds(providername).index(default_model)
	
	return idx

def getmodelIds(providername):
	models =[]
	bedrock = client()
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName']:
			models.append(model['modelId'])
			
	return models

def getmodelIds_claude3(providername='Anthropic'):
	models =[]
	bedrock = client()
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName'] and "IMAGE" in model['inputModalities']:
			models.append(model['modelId'])
			
	return models

def update_parameters(suffix,**args):
	for key in args:
		st.session_state[suffix][key] = args[key]
	return st.session_state[suffix]


def prompt_box(key,provider,model,maxTokenCount=1024,temperature=0.1,topP=0.9,topK=100,context=None):
	response = ''
	with st.container(border=True):
		prompt = st.text_area("Enter your prompt here",
							  height=100,
							  key=f"Q{key}")
		submit = st.button("Submit", type="primary", key=f"S{key}")

	if submit:
		if context is not None:
			prompt = context + "\n\n" + prompt
			match provider:
				case "Amazon":
					prompt = titan_generic(prompt)
				case "Anthropic":
					prompt = claude_generic(prompt)
				case "Meta":
					prompt = llama2_generic(prompt)
				case "Mistral":
					prompt = mistral_generic(prompt)
				case _:
					prompt = prompt
			
		with st.spinner("Generating..."):
			response = invoke_model(
				runtime_client(), 
				prompt, 
				model=model, 
				max_tokens=maxTokenCount, 
				temperature=temperature, 
				top_p=topP,
				top_k=topK)
				
	return response

list_providers = ['Amazon','Anthropic','AI21','Cohere','Meta','Mistral']

def invoke_model(client, prompt, model, 
	accept = 'application/json', content_type = 'application/json',
	max_tokens  = 512, temperature = 0.1, top_p = 0.9, top_k = 100, stop_sequences = [],
	count_penalty = 0, presence_penalty = 0, frequency_penalty = 0, return_likelihoods = 'NONE'):
	# default response
	output = ''
	# identify the model provider
	provider = model.split('.')[0] 
	# InvokeModel
	if ('claude-3' in model.split('.')[1] ): 
		input = {
			'max_tokens': max_tokens,
			'stop_sequences': stop_sequences,
			'temperature': temperature,
			'top_p': top_p,
			'top_k': top_k,
			"anthropic_version": "bedrock-2023-05-31",
			"messages": [{"role": "user", "content": prompt}]
		}
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept, contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body.get('content')[0]['text']
 
	elif (provider == 'anthropic'): 
		input = {
			'prompt': prompt,
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
			'inputText': prompt,
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
			'prompt': prompt,
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

# dataset = load_jsonl('mistral.jsonl')
# initsessionkeys(dataset[0])
# update_options(dataset,item_num=0)


