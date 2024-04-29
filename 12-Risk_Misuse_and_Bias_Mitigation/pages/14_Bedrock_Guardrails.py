import boto3
import json
import streamlit as st

st.set_page_config(
	page_title="Bedrock Guardrails",
	page_icon=":rocket:",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.subheader("Bedrock Guardrails")
st.markdown("""Guardrails for Amazon Bedrock evaluates user inputs and FM responses based on use case specific policies, \
and provides an additional layer of safeguards regardless of the underlying FM. Guardrails can be applied across all large \
language models (LLMs) on Amazon Bedrock, including fine-tuned models. Customers can create multiple guardrails, \
each configured with a different combination of controls, and use these guardrails across different applications and use cases.""")


client = boto3.client('bedrock', region_name='us-east-1')
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')


guardrail = client.list_guardrails(
	guardrailIdentifier='efzsheyu2yvm')['guardrails'][-1]


prompt2 = """Please summarize the below call center transcript. Put the name, email and the booking ID to the top:\n
Agent: Welcome to ABC company. How can I help you today?
Customer: I want to cancel my hotel booking.\n
Agent: Sure, I can help you with the cancellation. Can you please provide your booking ID?
Customer: Yes, my booking ID is 550e8408.\n
Agent: Thank you. Can I have your name and email for confirmation?
Customer: My name is Jane Doe and my email is jane.doe@gmail.com\n
Agent: Thank you for confirming. I will go ahead and cancel your reservation.
"""


with st.container(border=True):
	st.write(":orange[Guardrail Configuration:]")
	prompt_filter,  response_filter, profanity_topic = st.columns([0.3,0.3,0.4])

	with prompt_filter:
		with st.popover("Filter strengths for prompts", use_container_width=True):
			p_hate = st.select_slider("Hate", options=["None", "Low", "Medium", "High"], value="High" , key='p_hate')
			p_insults = st.select_slider("Insults", options=["None", "Low", "Medium", "High"], value="High", key='p_insults')
			p_sexual = st.select_slider("Sexual", options=["None", "Low", "Medium", "High"], value="High", key='p_sexual')
			p_violence = st.select_slider("Violence", options=["None", "Low", "Medium", "High"], value="High", key='p_violence')
			p_misconduct = st.select_slider("Misconduct", options=["None", "Low", "Medium", "High"], value="High", key='p_misconduct')
			p_prompt_attack = st.select_slider("Prompt Attack ", options=["None", "Low", "Medium", "High"], value="High", key='p_prompt_attack')
	with response_filter:
		with st.popover("Filter strengths for responses", use_container_width=True):
			r_hate = st.select_slider("Hate", options=["None", "Low", "Medium", "High"], value="High")
			r_insults = st.select_slider("Insults", options=["None", "Low", "Medium", "High"], value="High")
			r_sexual = st.select_slider("Sexual", options=["None", "Low", "Medium", "High"], value="High")
			r_violence = st.select_slider("Violence", options=["None", "Low", "Medium", "High"], value="High")
			r_misconduct = st.select_slider("Misconduct", options=["None", "Low", "Medium", "High"], value="High") 
	with profanity_topic:
		profanity_filter = st.toggle("Profanity Filter", value=True)
   
   

	denied_topic, word_filter, pii_filter = st.columns([0.4,0.3,0.3])
	with denied_topic:
		with st.popover("Denied Topic", use_container_width=True):
			topic_name = st.text_input('Name', value="Fiduciary Advice")
			definition = st.text_area("Definition for topic", value='Providing personalized advice or recommendations on managing financial assets, investments, or trusts in a fiduciary capacity or assuming related obligations and liabilities.')
			st.write('Examples:')
			example1 = st.text_input("ex1",value='What stocks should I invest in for my retirement?', label_visibility="hidden")
			example2 = st.text_input("ex2",value='Is it a good idea to put my money in a mutual fund?', label_visibility="hidden")
			example3 = st.text_input("ex3",value='How should I allocate my 401(k) investments?', label_visibility="hidden")
			example4 = st.text_input("ex4",value='What type of trust fund should I set up for my children?', label_visibility="hidden")
			example5 = st.text_input("ex5",value='Should I hire a financial advisor to manage my investments?', label_visibility="hidden")
	with word_filter:
		with st.popover("Word Filter", use_container_width=True):
			example1 = st.text_input("ex1",value='fiduciary advice', label_visibility="hidden")
			example2 = st.text_input("ex2",value='investment recommendations', label_visibility="hidden")
			example3 = st.text_input("ex3",value='stock picks', label_visibility="hidden")
			example4 = st.text_input("ex4",value='financial planning guidance', label_visibility="hidden")
			example5 = st.text_input("ex5",value='retirement fund suggestions', label_visibility="hidden")  
			example6 = st.text_input("ex6",value='wealth management tips', label_visibility="hidden")
			example7 = st.text_input("ex7",value='trust fund setup', label_visibility="hidden")
			example8 = st.text_input("ex8",value='investment strategy', label_visibility="hidden")
			example9 = st.text_input("ex9",value='financial advisor recommendations', label_visibility="hidden")
			example0 = st.text_input("ex0",value='portfolio allocation advice', label_visibility="hidden")
	with pii_filter:
		with st.popover("PII Reduction", use_container_width=True):
			type, behavior = st.columns(2)
			with type:
				st.write("PII type")
				email = st.text_input("ex1",value='Email', label_visibility="hidden")
				phone = st.text_input("ex2",value='Phone', label_visibility="hidden")
				name = st.text_input("ex3",value='Name', label_visibility="hidden")
				account = st.text_input("ex4",value='Account Number', label_visibility="hidden")
				booking = st.text_input("ex5",value='Booking ID', label_visibility="hidden")
			with behavior:
				st.write("Guardrail behavior")
				b_email = st.radio("ex1",['Mask', 'Block'], label_visibility="hidden",horizontal=False)
				b_phone = st.radio("ex2",['Mask', 'Block'], label_visibility="hidden",horizontal=False)
				b_name = st.radio("ex3",['Mask', 'Block'], label_visibility="hidden",horizontal=False)
				b_account = st.radio("ex4",['Mask', 'Block'], label_visibility="hidden",horizontal=False)
				b_booking= st.radio("ex5",['Mask', 'Block'], label_visibility="hidden",horizontal=False)

def getmodelId(providername):
	model_mapping = {
		"Amazon": "amazon.titan-tg1-large",
		"Anthropic": "anthropic.claude-3-haiku-20240307-v1:0",
		"AI21": "ai21.j2-ultra-v1",
		'Cohere': "cohere.command-text-v14",
		'Meta': "meta.llama2-70b-chat-v1",
		"Mistral": "mistral.mistral-large-2402-v1:0",
	}

	return model_mapping[providername]


def getmodelIds(providername):
	models = []
	bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
	available_models = bedrock.list_foundation_models()

	for model in available_models['modelSummaries']:
		if providername in model['providerName']:
			models.append(model['modelId'])

	return models

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

def invoke_model(client, prompt, model, 
	accept = 'application/json', content_type = 'application/json',
	max_tokens  = 512, temperature = 1.0, top_p = 1.0, top_k = 200, stop_sequences = [],
	count_penalty = 0, presence_penalty = 0, frequency_penalty = 0, return_likelihoods = 'NONE',
	guardrailIdentifier=guardrail['id'],
	guardrailVersion=guardrail['version'],
	trace="ENABLED" ):
	# default response
	output = ''
	response_body = ''
	model_output = None
	# identify the model provider
	provider = model.split('.')[0] 
	# InvokeModel
	if (provider == 'anthropic'): 
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
		response = client.invoke_model(body=body, modelId=model, accept=accept, contentType=content_type,
                                 guardrailIdentifier=guardrailIdentifier,guardrailVersion=guardrailVersion,trace=trace)
		response_body = json.loads(response.get('body').read())
		if response_body["amazon-bedrock-trace"]["guardrail"].get("modelOutput") is not None:
			model_output = response_body["amazon-bedrock-trace"]["guardrail"]["modelOutput"][0]
			model_output = json.loads(model_output)["content"][0]['text']
		output = response_body.get('content')[0]['text']
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
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type,
                                 guardrailIdentifier=guardrailIdentifier,guardrailVersion=guardrailVersion,trace=trace)
		response_body = json.loads(response.get('body').read())
		if response_body["amazon-bedrock-trace"]["guardrail"].get("modelOutput") is not None:
			model_output = response_body["amazon-bedrock-trace"]["guardrail"]["modelOutput"][0]
			model_output = json.loads(model_output)["prompt"]['text']
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
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type,
                                 guardrailIdentifier=guardrailIdentifier,guardrailVersion=guardrailVersion,trace=trace)
		response_body = json.loads(response.get('body').read())
		if response_body["amazon-bedrock-trace"]["guardrail"].get("modelOutput") is not None:
			model_output = response_body["amazon-bedrock-trace"]["guardrail"]["modelOutput"][0]
			model_output = json.loads(model_output)['results'][0]['outputText']
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
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type,
                                 guardrailIdentifier=guardrailIdentifier,guardrailVersion=guardrailVersion,trace=trace)
		response_body = json.loads(response.get('body').read())
		if response_body["amazon-bedrock-trace"]["guardrail"].get("modelOutput") is not None:
			model_output = response_body["amazon-bedrock-trace"]["guardrail"]["modelOutput"][0]
			model_output = json.loads(model_output)['generations'][0]['text']
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
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type,
                                 guardrailIdentifier=guardrailIdentifier,guardrailVersion=guardrailVersion,trace=trace)
		response_body = json.loads(response.get('body').read())
		if response_body["amazon-bedrock-trace"]["guardrail"].get("modelOutput") is not None:
			model_output = response_body["amazon-bedrock-trace"]["guardrail"]["modelOutput"][0]
			model_output = json.loads(model_output)['generation']
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
		response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type,
                                 guardrailIdentifier=guardrailIdentifier,guardrailVersion=guardrailVersion,trace=trace)
		response_body = json.loads(response.get('body').read())
		if response_body["amazon-bedrock-trace"]["guardrail"].get("modelOutput") is not None:
			model_output = response_body["amazon-bedrock-trace"]["guardrail"]["modelOutput"][0]
			model_output = json.loads(model_output).get('outputs')[0].get('text')
		output = response_body.get('outputs')[0].get('text')
	return output, response_body, model_output


def prompt_box(prompt, height, key):
	with st.form(f'form-{key}'):
		prompt_data = st.text_area(":orange[User Prompt:]", value=prompt, height=height)
		submit = st.form_submit_button("Submit", type='primary')

	return submit, prompt_data



def get_output(prompt, model, max_tokens, temperature, top_p):
	with st.spinner("Thinking..."):
		output, response_body, model_output = invoke_model(
			client=bedrock_runtime,
			prompt=prompt,
			model=model,
			temperature=temperature,
			top_p=top_p,
			max_tokens=max_tokens,
		)
		with st.expander("See Guardrail Trace:"):
			st.json(response_body)
   
		if model_output is not None:
			st.write(":orange[Model Response:]")
			st.info(model_output)
   
		st.write(":orange[Final Response:]")
		st.success(output)


prompt_col, param_col = st.columns([0.7,0.3])

with param_col:
	with st.container(border=True):
		provider = st.selectbox(
			'Provider:', ['Amazon', 'Anthropic', 'AI21', 'Cohere', 'Meta', 'Mistral'], index=1)
		model = st.selectbox('model', getmodelIds(provider),
							 index=getmodelIds(provider).index(getmodelId(provider)))

		temperature = st.slider('temperature', min_value=0.0,
								max_value=1.0, value=0.1, step=0.1)
		top_p = st.slider('top_p', min_value=0.0,
						  max_value=1.0, value=0.9, step=0.1)
		max_tokens = st.number_input(
			'max_tokens', min_value=50, max_value=4096, value=1024, step=1) 

with prompt_col:

	sample1, sample2, sample3 = st.tabs(['Denied Topic','PII Reduction', 'Try your own prompt'])

	with sample1:
		prompt1 = "How should I invest for my retirement? I want to be able to generate $5,000 a month"
		submit, prompt_data = prompt_box(prompt1, 100, 1)
		if submit:
			get_output(prompt_data, model, max_tokens=max_tokens,
					   temperature=temperature, top_p=top_p)
		
	with sample2:
		submit, prompt_data = prompt_box(prompt2, 300, 2)
		if submit:
			get_output(prompt_data, model, max_tokens=max_tokens,
					   temperature=temperature, top_p=top_p)

	with sample3:
		prompt3 = None
		submit, prompt_data = prompt_box(prompt3, 100, 3)
		if submit and prompt_data is not None:
			get_output(prompt_data, model, max_tokens=max_tokens,
					   temperature=temperature, top_p=top_p)
    
    
    
    
