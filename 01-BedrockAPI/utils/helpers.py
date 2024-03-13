import streamlit as st
import jsonlines
import json
import utils
import base64
from io import BytesIO
from random import randint
from jinja2 import Environment, FileSystemLoader
from utils import bedrock_runtime_client

def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d

def initsessionkeys(dataset):
    for key in dataset.keys():
        # print(key)
        if key not in st.session_state:
            st.session_state[key] = dataset[key]
    # print(st.session_state)
    return st.session_state

def update_options(dataset,item_num):
    for key in dataset[item_num]:
        if key in ["model","temperature","top_p","top_k","max_tokens"]:
            continue
        else:
            st.session_state[key] = dataset[item_num][key]
        # print(key, dataset[item_num][key])

def load_options(dataset,item_num):    
    # dataset = load_jsonl('mistral.jsonl')
    st.write("Prompt:",dataset[item_num]["prompt"])
    if "negative_prompt" in dataset[item_num].keys():
        st.write("Negative Prompt:", dataset[item_num]["negative_prompt"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(dataset,item_num))  


def create_two_tabs(dataset):
    tab1, tab2 = st.tabs(["Prompt1", "Prompt2"])
    with tab1:
        load_options(dataset,item_num=0)
    with tab2:
        load_options(dataset,item_num=1)
 
def create_three_tabs(dataset):
    tab1, tab2, tab3 = st.tabs(["Prompt1", "Prompt2", "Prompt3"])
    with tab1:
        load_options(dataset,item_num=0)
    with tab2:
        load_options(dataset,item_num=1) 
    with tab3:
        load_options(dataset, item_num=2)

def create_four_tabs(dataset):
    tab1, tab2, tab3, tab4 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)

def create_five_tabs(dataset):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)
    with tab5:
        load_options(dataset, item_num=4)

def create_six_tabs(dataset):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)
    with tab5:
        load_options(dataset, item_num=4)
    with tab6:
        load_options(dataset, item_num=5)

def create_seven_tabs(dataset):
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6", "Prompt7"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)
    with tab5:
        load_options(dataset, item_num=4)
    with tab6:
        load_options(dataset, item_num=5)
    with tab7:
        load_options(dataset, item_num=6)

def create_tabs(dataset):
    if len(dataset) == 2:
        create_two_tabs(dataset)
    elif len(dataset) == 3:
        create_three_tabs(dataset)
    elif len(dataset) == 4:
        create_four_tabs(dataset)
    elif len(dataset) == 5:
        create_five_tabs(dataset)
    elif len(dataset) == 6:
        create_six_tabs(dataset)
    elif len(dataset) == 7:
        create_seven_tabs(dataset)

def update_parameters(**args):
    if 'temperature' in args:
        st.session_state['temperature'] = args['temperature']
    if 'top_p' in args:
        st.session_state['top_p'] = args['top_p']
    if 'top_k' in args:
        st.session_state['top_k'] = args['top_k']
    if 'model' in args:
        st.session_state['model'] = args['model']
    if 'max_tokens' in args:
        st.session_state['max_tokens'] = args['max_tokens']
    if 'cfg_scale' in args:
        st.session_state['cfg_scale'] = args['cfg_scale']
    if 'seed' in args:
        st.session_state['seed'] = args['seed']
    if 'steps' in args:
        st.session_state['steps'] = args['steps']
    if 'quality' in args:
        st.session_state['quality'] = args['quality']
    if 'width' in args:
        st.session_state['width'] = args['width']
    if 'height' in args:
        st.session_state['height'] = args['height']
    if 'numberOfImages' in args:
        st.session_state['numberOfImages'] = args['numberOfImages']


def tune_parameters(provider, index=0,region='us-east-1'):
    st.subheader("Parameters")

    with st.form(key ='Form1'):
        models = utils.get_models(provider,region=region)
        model  = st.selectbox('model', models,index=index)
        temperature = st.slider('temperature',min_value = 0.0, max_value = 1.0, value = st.session_state['temperature'], step = 0.1)
        top_p = st.slider('top_p',min_value = 0.0, max_value = 1.0, value = st.session_state['top_p'], step = 0.1)
        max_tokens = st.number_input('max_tokens',min_value = 50, max_value = 4096, value = st.session_state['max_tokens'], step = 1)
        if provider in ['Anthropic','Cohere']:
            top_k = st.slider('top_k',min_value = 0, max_value = 300, value = st.session_state['top_k'], step = 1)
            params = {"model":model , "temperature":temperature, "top_p":top_p, "top_k":top_k ,"max_tokens":max_tokens}
        else:
            params = {"model":model , "temperature":temperature, "top_p":top_p,"max_tokens":max_tokens}
        st.form_submit_button(label = 'Tune Parameters', on_click=update_parameters, kwargs=(params)) 


def image_parameters(provider, index=0,region='us-east-1'):
    st.subheader("Parameters")
    with st.form(key ='Form2'):
        models = utils.get_models(provider,region=region)
        st.session_state['model']  = st.selectbox('model', models,index=index)
        st.session_state['cfg_scale'] = st.number_input('cfg_scale',value = st.session_state['cfg_scale'])
        st.session_state['seed']=st.number_input('seed',value = st.session_state['seed'])
        if provider == "Stability AI":
            st.session_state['steps']=st.number_input('steps',value = st.session_state['steps'])
            params = {"model":st.session_state['model'] , "cfg_scale":st.session_state['cfg_scale'], "seed":st.session_state['seed'],"steps":st.session_state['steps']}
        else:
            st.session_state['quality']=st.radio('quality',["premium", "standard"], horizontal=True)
            st.session_state['width']=st.number_input('width',value = st.session_state['width'])
            st.session_state['height']=st.number_input('height',value = st.session_state['height'])
            # st.session_state['numberOfImages']=st.number_input('numberOfImages',value = st.session_state['numberOfImages'])
            params = {"model":st.session_state['model'] ,"cfg_scale":st.session_state['cfg_scale'], "seed":st.session_state['seed'],"quality":st.session_state['quality'],
                      "width":st.session_state['width'],"height":st.session_state['height'],"numberOfImages":st.session_state['numberOfImages']}
        st.form_submit_button(label = 'Tune Parameters', on_click=update_parameters, kwargs=(params)) 

def invoke_model(client, prompt, model, 
    accept = 'application/json', content_type = 'application/json',
    max_tokens  = 512, temperature = 1.0, top_p = 1.0, top_k = 250, stop_sequences = [],
    count_penalty = 0, presence_penalty = 0, frequency_penalty = 0, return_likelihoods = 'NONE'):
    # default response
    output = ''
    # identify the model provider
    provider = model.split('.')[0] 
    # InvokeModel
    if (provider == 'anthropic'): 
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
    # return
    return output

# dataset = load_jsonl('mistral.jsonl')
# initsessionkeys(dataset[0])
# update_options(dataset,item_num=0)


#get the stringified request body for the InvokeModel API call
def get_titan_image_generation_request_body(prompt, negative_prompt,numberOfImages,quality, height, width,cfgScale,seed):
    
    body = { #create the JSON payload to pass to the InvokeModel API
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": negative_prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": numberOfImages,  # Number of images to generate
            "quality": quality,
            "height": height,
            "width": width,
            "cfgScale": cfgScale,
            "seed": seed
        }
    }
    
    # if negative_prompt:
    #     body['textToImageParams']['negativeText'] = negative_prompt
    
    return json.dumps(body)


#get a BytesIO object from the Titan Image Generator response
def get_titan_response_image(response):

    response = json.loads(response.get('body').read())
    
    images = response.get('images')
    
    image_data = base64.b64decode(images[0])

    return BytesIO(image_data)


#generate an image using Amazon Titan Image Generator
def get_image_from_model(prompt_content, negative_prompt, numberOfImages, quality, height, width, cfgScale, seed):

    bedrock = bedrock_runtime_client()
    
    body = get_titan_image_generation_request_body(prompt=prompt_content, negative_prompt=negative_prompt,numberOfImages=numberOfImages,quality=quality, height=height, width=width,cfgScale=cfgScale,seed=seed)
    
    response = bedrock.invoke_model(body=body, modelId="amazon.titan-image-generator-v1", contentType="application/json", accept="application/json")
    
    output = get_titan_response_image(response)
    
    return output


def reset_session():
    def form_callback():
        for key in st.session_state.keys():
            del st.session_state[key]


    st.sidebar.button(label='Reset Session', on_click=form_callback)


def render_titan_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], top_p=st.session_state['top_p'],
        model = st.session_state['model'])
    return output

def render_titan_image_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], quality=st.session_state['quality'], 
        height=st.session_state['height'], width=st.session_state['width'],
        cfgScale=st.session_state['cfg_scale'], seed=st.session_state['seed'],
        negative_prompt=st.session_state['negative_prompt'], numberOfImages=st.session_state['numberOfImages'],
        model = st.session_state['model'])
    return output

def render_claude_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], 
        max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], 
        top_p=st.session_state['top_p'],
        top_k = st.session_state['top_k'],
        model = st.session_state['model'])
    return output

def render_cohere_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], 
        max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], 
        top_p=st.session_state['top_p'],
        top_k = st.session_state['top_k'],
        model = st.session_state['model'])
    return output

def render_meta_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], top_p=st.session_state['top_p'],
        model = st.session_state['model'])
    return output

def render_mistral_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], top_p=st.session_state['top_p'],
        model = st.session_state['model'])
    return output

def render_stabilityai_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], cfg_scale=st.session_state['cfg_scale'], 
        seed=st.session_state['seed'], steps=st.session_state['steps'],
        model = st.session_state['model'])
    return output