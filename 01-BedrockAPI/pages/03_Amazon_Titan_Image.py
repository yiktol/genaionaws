import streamlit as st
import utils.helpers as helpers
from io import BytesIO
from random import randint
import utils as u

u.set_page_config()

helpers.reset_session()

bedrock_runtime = u.bedrock_runtime_client()

dataset = helpers.load_jsonl('data/titan_image.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6,0.4])


with text:

    st.title('Amazon Titan Image')
    st.write("""Titan Image Generator G1 is an image generation model. \
It generates images from text, and allows users to upload and edit an existing image. \
Users can edit an image with a text prompt (without a mask) or parts of an image with an image mask, or extend the boundaries of an image with outpainting. \
It can also generate variations of an image.""")

    with st.expander("See Code"):
        st.code(helpers.render_titan_image_code('titan_image.jinja'),language="python")
    
    st.subheader("Image parameters")
    
    with st.form("form1"):
        prompt_text = st.text_area("What you want to see in the image:",  height = st.session_state['prompt_height'], key = "prompt", help="The prompt text")
        negative_prompt = st.text_area("What shoud not be in the image:", height = st.session_state['n_prompt_height'], key="negative_prompt", help="The negative prompt")
        generate_button = st.form_submit_button("Generate", type="primary")

    if generate_button:

        st.subheader("Result")
        with st.spinner("Drawing..."):
            generated_image = helpers.get_image_from_model(
                prompt_content = prompt_text, 
                negative_prompt = negative_prompt,
                numberOfImages = st.session_state['numberOfImages'],
                quality = st.session_state['quality'], 
                height = st.session_state['height'], 
                width = st.session_state['width'], 
                cfgScale = st.session_state['cfg_scale'], 
                seed = st.session_state['seed']
            )
        st.image(generated_image)

with code:
    helpers.image_parameters('Amazon', index=2,region='us-east-1')

    st.subheader('Prompt Examples:')   
    with st.container(border=True):
        helpers.create_tabs(dataset)
