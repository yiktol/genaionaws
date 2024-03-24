import streamlit as st
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.claude3 as claude3
from IPython.display import Image
import base64


stlib.set_page_config()

suffix = 'claude3'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

if "image" not in st.session_state:
    st.session_state.image = "images/sg_skyline.jpg"

stlib.initsessionkeys(claude3.params,suffix)

with open(st.session_state.image, "rb") as image_file:
    binary_data = image_file.read()
    base_64_encoded_data = base64.b64encode(binary_data)
    base64_string = base_64_encoded_data.decode('utf-8')

text, code = st.columns([0.6,0.4])


with text:
    st.title("Anthropic Claude 3")
    st.write("""Anthropic's Claude family of models - Haiku, Sonnet, and Opus - allow customers to choose the exact combination of intelligence, speed, and cost that suits their business needs. \
Claude 3 Opus, the company's most capable model, has set a market standard on benchmarks. \
All of the latest Claude models have vision capabilities that enable them to process and analyze image data, meeting a growing demand for multimodal AI systems that can handle diverse data formats. While the family offers impressive performance across the board, Claude 3 Haiku is one of the most affordable and fastest options on the market for its intelligence category.""")

    # with st.expander("See Code"): 
    #     st.code(claude3.render_claude_code('claude.jinja',suffix),language="python")
    st.image(st.session_state.image)

    # Define prompt and model parameters
    prompt_input = "Write a python code that list all countries."
    
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = 100,
            value = "Describe the image"
        )
        submit = st.form_submit_button("Submit", type='primary')
        
    if prompt_data and submit:
        with st.spinner("Generating..."):
            response = claude3.invoke_model(client=bedrock.runtime_client(), 
                                            prompt=prompt_data,
                                            model=st.session_state[suffix]['model'], 
                                            max_tokens  = st.session_state[suffix]['max_tokens'], 
                                            temperature = st.session_state[suffix]['temperature'], 
                                            top_p = st.session_state[suffix]['top_p'],
                                            top_k = st.session_state[suffix]['top_k'],
                                            media_type="image/jpeg",
        		                            image_data=base64_string)

            st.write("### Answer")
            st.info(response)

with code:

    claude3.tune_parameters('Anthropic',suffix,index=6)
