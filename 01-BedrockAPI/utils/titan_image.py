import streamlit as st
import jsonlines
import json
import boto3
import base64
from io import BytesIO
from random import randint
from jinja2 import Environment, FileSystemLoader
from utils.bedrock import get_models


params = {
    "cfg_scale":8,
    "seed":randint(10,20000),
    "quality":"premium",
    "width":1024,
    "height":1024,
    "numberOfImages":1,
    "model":"amazon.titan-image-generator-v1",
    }


def render_titan_image_code(templatePath,suffix):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state[suffix]['prompt'], 
        quality=st.session_state[suffix]['quality'], 
        height=st.session_state[suffix]['height'], 
        width=st.session_state[suffix]['width'],
        cfgScale=st.session_state[suffix]['cfg_scale'], 
        seed=st.session_state[suffix]['seed'],
        negative_prompt=st.session_state[suffix]['negative_prompt'], 
        numberOfImages=st.session_state[suffix]['numberOfImages'],
        model = st.session_state[suffix]['model'])
    return output


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

def image_parameters(provider, suffix, index=0,region='us-east-1'):
    st.subheader("Parameters")
    with st.container(border=True):
        models = get_models(provider,region=region)
        model  = st.selectbox('model', models,index=index)
        cfg_scale= st.number_input('cfg_scale',value = 8)
        seed=st.number_input('seed',value = randint(10,20000))
        quality=st.radio('quality',["premium", "standard"], horizontal=True)
        width=st.number_input('width',value = 1024)
        height=st.number_input('height',value = 1024)
        numberOfImages=st.number_input('numberOfImages',value = 1)
        params = {"model":model ,"cfg_scale":cfg_scale, "seed":seed,"quality":quality,
                    "width":width,"height":height,"numberOfImages":numberOfImages}
        st.button(label = 'Tune Parameters', on_click=update_parameters, args=(suffix,), kwargs=(params)) 



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

    bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    
    body = get_titan_image_generation_request_body(prompt=prompt_content, negative_prompt=negative_prompt,numberOfImages=numberOfImages,quality=quality, height=height, width=width,cfgScale=cfgScale,seed=seed)
    
    response = bedrock.invoke_model(body=body, modelId="amazon.titan-image-generator-v1", contentType="application/json", accept="application/json")
    
    output = get_titan_response_image(response)
    
    return output

