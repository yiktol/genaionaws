
import streamlit as st
import json
import boto3


st.set_page_config(
	page_title="Gen AI in Practice",
	page_icon=":rocket:",
	layout="wide",
	initial_sidebar_state="expanded",
)

client = boto3.client(service_name='bedrock-runtime',region_name='us-east-1' )

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

def invoke_model(client, prompt, model, 
	accept = 'application/json', content_type = 'application/json',
	max_tokens  = 512, temperature = 1.0, top_p = 1.0, top_k = 200, stop_sequences = [],
	count_penalty = 0, presence_penalty = 0, frequency_penalty = 0, return_likelihoods = 'NONE'):
	# default response
	output = ''
	# identify the model provider
	provider = model.split('.')[0] 
	# InvokeModel
	if (provider == 'anthropic'): 
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


text, code = st.columns([0.6,0.4])

prompt1 = """Meet Carbon Maps, a new French startup that raised $4.3 million (€4 million) just a few weeks after its inception. The company is building a software-as-a-service platform for the food industry so that they can track the environmental impact of each of their products in their lineup. The platform can be used as a basis for eco ratings. \
While there are quite a few carbon accounting startups like Greenly, Sweep, Persefoni and Watershed, Carbon Maps isn't an exact competitor as it doesn't calculate a company's carbon emissions as a whole. It doesn't focus on carbon emissions exclusively either. Carbon Maps focuses on the food industry and evaluates the environmental impact of products — not companies. \
Co-founded by Patrick Asdaghi, Jérémie Wainstain and Estelle Huynh, the company managed to raise a seed round with Breega and Samaipata — these two VC firms already invested in Asdaghi's previous startup, FoodChéri. \
FoodChéri is a full-stack food delivery company that designs its own meals and sells them directly to end customers with an important focus on healthy food. It also operates Seazon, a sister company for batch deliveries. The startup was acquired by Sodexo a few years ago. \
“On the day that I left, I started working on food and health projects again,” Asdaghi told me. “I wanted to make an impact, so I started moving up the supply chain and looking at agriculture.” \
And the good news is that Asdaghi isn't the only one looking at the supply chain of the food industry. In France, some companies started working on an eco-score with a public agency (ADEME) overseeing the project. It's a life cycle assessment that leads to a letter rating from A to E. \
While very few brands put these letters on their labels, chances are companies that have good ratings will use the eco-score as a selling point in the coming years. \
But these ratings could become even more widespread as regulation is still evolving. The European Union is even working on a standard — the Product Environmental Footprint (PEF). European countries can then create their own scoring systems based on these European criteria, meaning that food companies will need good data on their supply chains. \
“The key element in the new eco-score that's coming up is that there will be some differences within a product category because ingredients and farming methods are different,” Asdaghi said. “It's going to take into consideration the carbon impact, but also biodiversity, water consumption and animal welfare.” \
For instance, when you look at ground beef, it's extremely important to know whether farmers are using soy from Brazil or grass to feed cattle. \
“We don't want to create the ratings. We want to create the tools that help with calculations — a sort of SAP,” Asdaghi said. \
So far, Carbon Maps is working with two companies on pilot programs as it's going to require a ton of work to cover each vertical in the food industry. The startup creates models with as many criteria as possible to calculate the impact of each criteria. It uses data from standardized sources like GHG Protocol, IPCC, ISO 14040 and 14044. \
The company targets food brands because they design the recipes and select their suppliers. Eventually, Carbon Maps hopes that everybody across the supply chain is going to use its platform in one way or another. \
“You can't have a true climate strategy if you don't have some collaboration across the chain,” Asdaghi said. \

## 

Summarize the above text in 5 bullets."""


prompt2 = """Please precisely copy any email addresses from the following text and then write them in a table with index number.. Only write an email address if it's precisely spelled out in the input text. If there are no email addresses in the text, write "N/A". Do not say anything else.\n
"Phone Directory:
John Latrabe, 800-232-1995, john909709@geemail.com
Josie Lana, 800-759-2905, josie@josielananier.com
Keven Stevens, 800-980-7000, drkevin22@geemail.com 
Phone directory will be kept up to date by the HR manager." 
"""

prompt3 = """I'd like you to translate this paragraph into English:

白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
"""

prompt4 = """Write an informational article for children about how birds fly.  \
Compare how birds fly to how airplanes fly.  \
Make sure to use the word "Thrust" at least three times.
"""

prompt5 = """Here is some text. We want to remove all personally identifying information from this text and replace it with XXX. It's very important that names, phone numbers, and email addresses, gets replaced with XXX. 
Here is the text, inside <text></text> XML tags\n
<text>
   Joe: Hi Hannah!
   Hannah: Hi Joe! Are you coming over?  
   Joe: Yup! Hey I, uh, forgot where you live." 
   Hannah: No problem! It's 4085 Paco Ln, Los Altos CA 94306.
   Joe: Got it, thanks!  
</text> \n
Please put your sanitized version of the text with PII removed in <response></response> XML tags 
"""


options = [{"id":1,"prompt": prompt1,"height":780},
		   {"id":2,"prompt": prompt2,"height":250},
		   {"id":3,"prompt": prompt3,"height":80},
		   {"id":4,"prompt": prompt4,"height":80},
		   {"id":5,"prompt": prompt5,"height":350},
		   ]


def prompt_box(prompt,key,height):
	with st.form(f"form-{key}"):
		prompt_data = st.text_area(
			":orange[Enter your prompt here:]",
			height = height,
			value=prompt,
			key=key)
		submit = st.form_submit_button("Submit", type="primary")
	
	return submit, prompt_data
		 

def get_output(client, prompt, model, max_tokens  = 512, temperature = 1.0, top_p = 1.0):
	with st.spinner("Thinking..."):
		output = invoke_model(client=client, 
							prompt=prompt, 
							model=model,
							temperature=temperature,
							top_p=top_p,
							max_tokens=max_tokens)
		st.write("Answer:")
		st.info(output)
	
list_providers = ['Amazon','Anthropic','AI21','Cohere','Meta','Mistral']

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

def getmodelIds(providername):
	models =[]
	bedrock = boto3.client(service_name='bedrock',region_name='us-east-1' )
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName']:
			models.append(model['modelId'])
			
	return models

prompt_col, paramaters = st.columns([0.7,0.3])

with paramaters:
	with st.form('Param-form'):
		provider = st.selectbox('provider', list_providers)
		models = getmodelIds(provider)
		model = st.selectbox(
			'model', models, index=models.index(getmodelId(provider)))
		# model = st.text_input('model', 'amazon.titan-tg1-large', disabled=True)
		temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
		# top_k=st.slider('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
		top_p=st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
		max_tokens=st.number_input('max_tokens',min_value = 50, max_value = 4096, value = 2048, step = 1)
		submitted = st.form_submit_button('Tune Parameters') 



with prompt_col:
	tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summarization", "Extraction", "Translation", "Generation", "Redaction"])
	with tab1:
		submit, prompt_data = prompt_box(options[0]["prompt"], options[0]["id"], options[0]["height"])
		if submit:
			get_output(client, prompt_data, model, max_tokens  = max_tokens, temperature = temperature, top_p = top_p)
	with tab2:
		submit, prompt_data = prompt_box(options[1]["prompt"], options[1]["id"], options[1]["height"])
		if submit:
			get_output(client, prompt_data, model, max_tokens  = max_tokens, temperature = temperature, top_p = top_p)
	with tab3:
		submit, prompt_data = prompt_box(options[2]["prompt"], options[2]["id"], options[2]["height"])
		if submit:
			get_output(client, prompt_data, model, max_tokens  = max_tokens, temperature = temperature, top_p = top_p)
	with tab4:
		submit, prompt_data = prompt_box(options[3]["prompt"], options[3]["id"], options[3]["height"])
		if submit:
			get_output(client, prompt_data, model, max_tokens  = max_tokens, temperature = temperature, top_p = top_p)
	with tab5:
		submit, prompt_data = prompt_box(options[4]["prompt"], options[4]["id"], options[4]["height"])
		if submit:
			get_output(client, prompt_data, model, max_tokens  = max_tokens, temperature = temperature, top_p = top_p)