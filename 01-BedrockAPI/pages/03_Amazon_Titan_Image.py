import streamlit as st
import utils.helpers as helpers
from io import BytesIO
from random import randint
import utils.helpers as helpers
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.titan_image as titan_image

stlib.set_page_config()

suffix = 'titan_image'
if suffix not in st.session_state:
    st.session_state[suffix] = {}


stlib.reset_session()

bedrock_runtime = bedrock.runtime_client()

dataset = helpers.load_jsonl('data/titan_image.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(titan_image.params,suffix)

text, code = st.columns([0.6,0.4])


with text:

    st.title('Amazon Titan Image')
    st.write("""Titan Image Generator G1 is an image generation model. \
It generates images from text, and allows users to upload and edit an existing image. \
Users can edit an image with a text prompt (without a mask) or parts of an image with an image mask, or extend the boundaries of an image with outpainting. \
It can also generate variations of an image.""")

    with st.expander("See Code"):
        st.code(titan_image.render_titan_image_code('titan_image.jinja',suffix),language="python")
    
    st.subheader("Image parameters")
    
    with st.form("form1"):
        prompt_text = st.text_area("What you want to see in the image:",  height = st.session_state[suffix]['prompt_height'], value = st.session_state[suffix]['prompt'], help="The prompt text")
        negative_prompt = st.text_area("What shoud not be in the image:", height = st.session_state[suffix]['n_prompt_height'], value = st.session_state[suffix]['negative_prompt'], help="The negative prompt")
        generate_button = st.form_submit_button("Generate", type="primary")

    if generate_button:

        st.subheader("Result")
        with st.spinner("Drawing..."):
            generated_image = titan_image.get_image_from_model(
                prompt_content = prompt_text, 
                negative_prompt = negative_prompt,
                numberOfImages = st.session_state[suffix]['numberOfImages'],
                quality = st.session_state[suffix]['quality'], 
                height = st.session_state[suffix]['height'], 
                width = st.session_state[suffix]['width'], 
                cfgScale = st.session_state[suffix]['cfg_scale'], 
                seed = st.session_state[suffix]['seed']
            )
        st.image(generated_image)

with code:
    titan_image.image_parameters('Amazon', suffix, index=2,region='us-east-1')

    st.subheader('Prompt Examples:')   
    with st.container(border=True):
        stlib.create_tabs(dataset,suffix)
