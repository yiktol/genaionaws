import streamlit as st
import utils.stlib as stlib
import utils.sdxl as sdxl

stlib.set_page_config()

suffix = 'sdxl'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

stlib.reset_session()

dataset = sdxl.load_jsonl('data/stabilityai.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(sdxl.params,suffix)

text, code = st.columns([0.6, 0.4])

with text:
    st.title("Stable Diffusion")
    st.write("Deep learning, text-to-image model used to generate detailed images conditioned on text descriptions, inpainting, outpainting, and generating image-to-image translations.")

    with st.expander("See Code"):
        st.code(sdxl.render_sdxl_image_code('sdxl.jinja',suffix), language="python")

    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area("What you want to see in the image:",
                                   height=st.session_state[suffix]['p_height'],value = st.session_state[suffix]['prompt'], help="The prompt text")
        negative_prompt = st.text_area("What you don't want to see in the image:",
                                       height=st.session_state[suffix]['n_height'], value = st.session_state[suffix]['negative_prompt'], help="The negative prompt text")
        submit = st.form_submit_button("Submit", type='primary')


    if prompt_data and submit:
        st.subheader("Result")
        with st.spinner("Drawing..."):
            generated_image = sdxl.get_image_from_model(
                prompt = prompt_data, 
                negative_prompt = negative_prompt,
                model=st.session_state[suffix]['model'],
                height = st.session_state[suffix]['height'], 
                width = st.session_state[suffix]['width'], 
                cfg_scale = st.session_state[suffix]['cfg_scale'], 
                seed = st.session_state[suffix]['seed'],
                steps = st.session_state[suffix]['steps'],
                
            )
        st.image(generated_image)


with code:

    sdxl.image_parameters("Stability AI",suffix, index=1, region='us-east-1')

    st.subheader('Prompt Examples:')
    with st.container(border=True):
        stlib.create_tabs(dataset,suffix)
