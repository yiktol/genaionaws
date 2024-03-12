import streamlit as st
import utils.helpers as helpers
from io import BytesIO
from random import randint
from utils import bedrock_runtime_client, set_page_config,titan_generic

set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.button(label='Reset Session', on_click=form_callback)

bedrock_runtime = bedrock_runtime_client()

dataset = helpers.load_jsonl('utils/titan_image.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6,0.4])


xcode = """import json
import boto3
import base64
from io import BytesIO
from random import randint

bedrock = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
    )

prompt = "Blue backpack on a table."
negative_prompt = ""

body = { 
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt,
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "quality": "premium",
        "height": 512,
        "width": 512,
        "cfgScale": 8.0,
        "seed": randint(0, 100000), 
    },
}

if negative_prompt:
    body['textToImageParams']['negativeText'] = negative_prompt

response = bedrock.invoke_model(
    body=body, 
    modelId="amazon.titan-image-generator-v1", 
    contentType="application/json", 
    accept="application/json")

response = json.loads(response.get('body').read())

images = response.get('images')

image_data = base64.b64decode(images[0])

image = BytesIO(image_data)
"""


with text:

    st.title('Amazon Titan Image')
    st.write("""Titan Image Generator G1 is an image generation model. \
It generates images from text, and allows users to upload and edit an existing image. \
Users can edit an image with a text prompt (without a mask) or parts of an image with an image mask, or extend the boundaries of an image with outpainting. \
It can also generate variations of an image.""")

    with st.expander("See Code"):
        st.code(xcode,language="python")
    
    st.subheader("Image parameters")
    
    with st.form("form1"):
        prompt_text = st.text_area("What you want to see in the image:",  height = st.session_state['ta_height'], key = "prompt", help="The prompt text")
        negative_prompt = st.text_area("What shoud not be in the image:", height = st.session_state['ta_height'], key="negative_prompt", help="The negative prompt")
        generate_button = st.form_submit_button("Generate", type="primary")

    if generate_button:

        st.subheader("Result")
        with st.spinner("Drawing..."):
            generated_image = helpers.get_image_from_model(
                prompt_content=prompt_text, 
                negative_prompt=negative_prompt,
                numberOfImages = st.session_state['numberOfImages'],
                quality = st.session_state['quality'], 
                height = st.session_state['height'], 
                width=st.session_state['width'], 
                cfgScale = st.session_state['cfg_scale'], 
                seed= randint(0, 100000)
            )
        st.image(generated_image)

with code:
    helpers.image_parameters('Amazon', index=2,region='us-east-1')

    st.subheader('Prompt Examples:')   
    with st.container(border=True):
        helpers.create_tabs(dataset)
