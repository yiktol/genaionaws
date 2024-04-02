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
import utils.claude3 as claude3
import utils.titan_image as titan_image
import utils.sdxl as sdxl


def tune_parameters(provider):
	match provider:
		case "Anthropic":
			claude2.initsessionkeys(claude2.params, 'claude2')
			params = claude2.tune_parameters()
		case "Amazon":
			titan.initsessionkeys(titan.params, 'titan')
			params = titan.tune_parameters()
		case "Claude 3":
			claude3.initsessionkeys(claude3.params, 'claude3')
			params = claude3.tune_parameters()
		case "Cohere":
			cohere.initsessionkeys(cohere.params, 'cohere')
			params = cohere.tune_parameters()
		case "AI21":
			jurassic.initsessionkeys(llama2.params, 'jurassic')          
			params = jurassic.tune_parameters()
		case "Mistral":
			mistral.initsessionkeys(mistral.params, 'mistral')
			params = mistral.tune_parameters()
		case "Meta":
			llama2.initsessionkeys(llama2.params, 'llama2')
			params = llama2.tune_parameters()
		case _:
			print("Provider not supported")
			return False
	return params

def image_parameters(provider):
	match provider:
		case 'Titan Image':          
			titan_image.initsessionkeys(titan_image.params,'titan-image')
			params = titan_image.image_parameters()
		case 'Stability AI':
			sdxl.initsessionkeys(sdxl.params,'sdxl')
			params = sdxl.image_parameters()
   
	return params

def image_model(provider):
	match provider:
		case 'Titan Image':          
			models = getmodelIds_titan_image()
		case 'Stability AI':
			models = getmodelIds(provider)
	model = st.selectbox(
		'model', models, index=models.index(getmodelId(provider)))  
 
	return model

def generate_image(provider,model, prompt,negative_prompt,**params):
	match provider:
		case 'Titan Image':   
			generated_image = titan_image.get_image_from_model(model,
				prompt_content = prompt, 
				negative_prompt = negative_prompt,
				**params
			)
			
		case 'Stability AI':
			generated_image = sdxl.get_image_from_model(model,
				prompt = prompt, 
				negative_prompt = negative_prompt,
				**params
				
			)
	return generated_image

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
	prompt = f"""User: {input_prompt}\n\nBot:"""
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
		"Titan Image": "amazon.titan-image-generator-v1",
		"Anthropic" : "anthropic.claude-v2:1",
		"Claude 3": "anthropic.claude-3-sonnet-20240229-v1:0",
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
	
	if providername == "Claude 3":
		idx = getmodelIds_claude3(providername).index(default_model)
	elif providername == "Titan Image":
		idx = getmodelIds_titan_image(providername).index(default_model)
	else:
		idx = getmodelIds(providername).index(default_model)
	
	return idx

def getmodelIds(providername):
	models =[]
	bedrock = client()
	available_models = bedrock.list_foundation_models()
	
	if providername == "Claude 3":
		for model in available_models['modelSummaries']:
			if 'claude-3' in model['modelId'].split('.')[1]:
				models.append(model['modelId'])
	else:
		for model in available_models['modelSummaries']:
			if providername in model['providerName']:
				models.append(model['modelId'])
			
	return models

def getmodelIds_claude3(providername='Anthropic'):
	models =[]
	bedrock = client()
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName'] and "anthropic.claude-3" in model['modelId']:
			models.append(model['modelId'])
			
	return models

def getmodelIds_titan_image(providername='Amazon'):
	models =[]
	bedrock = client()
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName'] and "amazon.titan-image" in model['modelId']:
			models.append(model['modelId'])
			
	return models

def update_parameters(suffix,**args):
	for key in args:
		st.session_state[suffix][key] = args[key]
	return st.session_state[suffix]


def prompt_box(key,provider,model,context=None,height=100,**params):
	response = ''
	with st.container(border=True):
		prompt_data = st.text_area("Enter your prompt here", value = context, 
                             height=height,
							  key=f"Q{key}")
		submit = st.button("Submit", type="primary", key=f"S{key}")

	if submit:
		if context is not None:
			prompt = context + "\n\n" + prompt_data
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
				**params)
				
	return response


list_providers = ['Amazon','Anthropic','AI21','Claude 3','Cohere','Meta','Mistral']

def invoke_model(client, prompt, model, accept = 'application/json', content_type = 'application/json',**params):
	# default response
	output = ''
	# identify the model provider
	provider = model.split('.')[0] 
	# InvokeModel
	if ('anthropic.claude-3' in model): 
		input = {
			"anthropic_version": "bedrock-2023-05-31",
			"messages": [{"role": "user", "content": prompt}]
		}

		input.update(params)
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept, contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body.get('content')[0]['text']
 
	elif ('anthropic.claude' in model): 
		input = {
			'prompt': prompt,
		}
		input.update(params)
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body['completion']
	elif (provider == 'ai21'): 
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
	elif (provider == 'amazon'): 
		input = {
			'inputText': prompt,
			'textGenerationConfig': params
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
		}
		input.update(params)
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		results = response_body['generations']
		for result in results:
			output = output + result['text']
	elif (provider == 'meta'): 
		input = {
			'prompt': prompt,
		}
		input.update(params)
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body['generation']
	elif (provider == 'mistral'): 
		input = {
			'prompt': prompt,
		}
		input.update(params)
		body=json.dumps(input)
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
		response_body = json.loads(response.get('body').read())
		output = response_body.get('outputs')[0].get('text')

	 
	return output	

# dataset = load_jsonl('mistral.jsonl')
# initsessionkeys(dataset[0])
# update_options(dataset,item_num=0)


def get_secret(secret_name):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    return get_secret_value_response['SecretString']