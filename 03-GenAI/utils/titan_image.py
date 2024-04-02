import streamlit as st
import jsonlines
import json
import boto3
import base64
from io import BytesIO
from random import randint
from jinja2 import Environment, FileSystemLoader
import uuid
import utils.helpers as u_bedrock


def getmodelId(providername):
    model_mapping = {
        "Amazon": "amazon.titan-tg1-large",
        "Amazon-Image": "amazon.titan-image-generator-v1",
        "Anthropic": "anthropic.claude-v2:1",
        "AI21": "ai21.j2-ultra-v1",
        'Cohere': "cohere.command-text-v14",
        'Meta': "meta.llama2-70b-chat-v1",
        "Mistral": "mistral.mixtral-8x7b-instruct-v0:1",
        "Stability AI": "stability.stable-diffusion-xl-v1"
    }

    return model_mapping[providername]


def getmodel_index(providername):

    default_model = getmodelId(providername)

    idx = getmodelIds(providername).index(default_model)

    return idx


def getmodelIds(providername):
    models = []
    bedrock = boto3.client(service_name='bedrock', region_name="us-east-1")
    available_models = bedrock.list_foundation_models()

    for model in available_models['modelSummaries']:
        if providername in model['providerName'] and "titan-image" in model['modelId']:
            models.append(model['modelId'])

    return models


def reset_session():
    def form_callback():
        for key in st.session_state.keys():
            del st.session_state[key]

    st.button(label='Reset', on_click=form_callback, key=uuid.uuid1())


def initsessionkeys(dataset, suffix):
    for key in dataset.keys():
        if key not in st.session_state[suffix]:
            st.session_state[suffix][key] = dataset[key]
    return st.session_state[suffix]


params = {
    "cfgScale": 8.0,
    "seed": randint(10, 20000),
    "quality": "premium",
    "width": 512,
    "height": 512,
    "numberOfImages": 1,
    "model": "amazon.titan-image-generator-v1",
}


def render_titan_image_code(templatePath, suffix):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state[suffix]['prompt'],
        quality=st.session_state[suffix]['quality'],
        height=st.session_state[suffix]['height'],
        width=st.session_state[suffix]['width'],
        cfgScale=st.session_state[suffix]['cfgScale'],
        seed=st.session_state[suffix]['seed'],
        negative_prompt=st.session_state[suffix]['negative_prompt'],
        numberOfImages=st.session_state[suffix]['numberOfImages'],
        model=st.session_state[suffix]['model'])
    return output


def update_parameters(suffix, **args):
    for key in args:
        st.session_state[suffix][key] = args[key]
    return st.session_state[suffix]


def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d


def image_parameters():
    size = ["512x512", "1024x1024", "768x768", "768x1152", "384x576",
            "1152x768", "576x384", "768x1280", "384x640", "1280x768", "640x384"]

    # models = getmodelIds(provider)
    # model = st.selectbox(
    #     'model', models, index=models.index(getmodelId('Amazon-Image')))
    cfgScale = st.slider(
        'cfgScale', value=8.0, min_value=1.1, max_value=10.0, step=1.0)
    seed = st.number_input('seed', value=10000)
    quality = st.radio('quality', ["premium", "standard"], horizontal=True)
    selected_size = st.selectbox('size', size, index=1)
    width = int(selected_size.split('x')[0])
    height = int(selected_size.split('x')[1])
    numberOfImages = st.selectbox('numberOfImages', [1], disabled=True)
    params = {
        	# "model": model,
              "cfgScale": cfgScale,
              "seed": seed,
              "quality": quality,
              "width": width,
              "height": height,
              "numberOfImages": numberOfImages
              }

    return params


# get the stringified request body for the InvokeModel API call
def get_titan_image_generation_request_body(prompt, negative_prompt=None, **params):

    body = {  # create the JSON payload to pass to the InvokeModel API
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
                    "negativeText": negative_prompt
        },
        "imageGenerationConfig": params
    }

    if negative_prompt:
        body['textToImageParams']['negativeText'] = negative_prompt

    return json.dumps(body)


# get a BytesIO object from the Titan Image Generator response
def get_titan_response_image(response):

    response = json.loads(response.get('body').read())

    images = response.get('images')

    image_data = base64.b64decode(images[0])

    return BytesIO(image_data)


# generate an image using Amazon Titan Image Generator
def get_image_from_model(model, prompt_content, negative_prompt=None, **params):

    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
    )

    body = get_titan_image_generation_request_body(prompt=prompt_content, negative_prompt=negative_prompt, **params)

    response = bedrock.invoke_model(body=body, modelId=model,
                                    contentType="application/json", accept="application/json")

    output = get_titan_response_image(response)

    return output
