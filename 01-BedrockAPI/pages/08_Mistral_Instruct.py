import json
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.mistral as mistral
import streamlit as st

stlib.set_page_config()

suffix = 'mistral'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

stlib.reset_session()

bedrock_runtime = bedrock.runtime_client(region='us-east-1')

dataset = mistral.load_jsonl('data/mistral.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(mistral.params,suffix)

text, code = st.columns([0.6,0.4])

with text:
    st.title('Mistral AI')
    st.markdown("""Mistral AI is a small creative team with high scientific standards. We make efficient, helpful and trustworthy AI models through ground-breaking innovations.
- A 7B dense Transformer, fast-deployed and easily customisable. Small, yet powerful for a variety of use cases. Supports English and code, and a 32k context window.
- A 7B sparse Mixture-of-Experts model with stronger capabilities than Mistral 7B. Uses 12B active parameters out of 45B total. Supports multiple languages, code and 32k context window.
                """)

    with st.expander("See Code"):
        st.code(mistral.render_mistral_code('instruct.jinja',suffix),language="python")

    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = st.session_state[suffix]['height'],
            value = st.session_state[suffix]["prompt"]
        )
        submit = st.form_submit_button("Submit", type='primary')

    if prompt_data and submit:
        with st.spinner("Generating..."):

            response = mistral.invoke_model(bedrock.runtime_client(), 
                                            prompt_data, 
                                            st.session_state[suffix]['model'], 
                                            max_tokens=st.session_state[suffix]['max_tokens'], 
                                            temperature=st.session_state[suffix]['temperature'], 
                                            top_p=st.session_state[suffix]['top_p'])

            st.write("### Answer")
            st.info(response)

with code:
    mistral.tune_parameters('Mistral',suffix, index=0,region='us-east-1')
    st.subheader('Prompt Examples:')   
    with st.container(border=True):
        stlib.create_tabs(dataset,suffix)



