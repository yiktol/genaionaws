import streamlit as st
import boto3
import json
from langchain.prompts import PromptTemplate

list_providers = ['Amazon','Anthropic','AI21','Cohere','Meta','Mistral']

#Create the connection to Bedrock
bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1', 
    
)

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




def prompt_box(key,provider,model,prompt,maxTokenCount=1024,temperature=0.1,topP=0.9,topK=100,context=None):
	response = ''
	with st.container(border=True):
		prompt = st.text_area("Enter your prompt here", value=prompt,
							  height=150,
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
	   


def getmodelparams(providername):
    model_mapping = {
        "Amazon" : {
            "maxTokenCount": 4096,
            "stopSequences": [], 
            "temperature": 0.5,
            "topP": 0.9
            },
        "Anthropic" : {
            "max_tokens_to_sample": 4096,
            "temperature": 0.9,
            "top_k": 250,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman"],
            },
        "AI21" : {
            "maxTokens": 4096,
            "temperature": 0.5,
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
            "temperature": 0.5,
            "p": 0.9,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        },
        "Meta":{ 
            "max_gen_len": 1024,
            "top_p": 0.9,
            "temperature": 0.8
        },
        "Mistral":{ 
            "max_tokens": 1024,
            "top_p": 0.9,
            "top_k": 50,
            "temperature": 0.1
        },
        "Stability":{ 
            "cfg_scale": 10,
            "seed": 0,
            "steps": 50,
            "width": 512,
            "height": 512
        }
    }
    
    return model_mapping[providername]

def getmodelId(providername):
    model_mapping = {
        "Amazon" : "amazon.titan-tg1-large",
        "Anthropic" : "anthropic.claude-v2:1",
        "AI21" : "ai21.j2-ultra-v1",
        "Cohere": "cohere.command-text-v14",
        "Meta": "meta.llama2-13b-chat-v1",
        "Mistral": "mistral.mistral-7b-instruct-v0:2"
    }
    
    return model_mapping[providername]

def set_page_config():
    
    st.set_page_config( 
    page_title="Prompt Engineering",  
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)
    
def bedrock_runtime_client():
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    return bedrock_runtime


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