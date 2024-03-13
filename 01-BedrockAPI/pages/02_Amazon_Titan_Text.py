import streamlit as st
import json
from utils import get_models, titan_generic, getmodelparams, set_page_config, bedrock_runtime_client, titan_generic
import utils.helpers as helpers
import streamlit as st

bedrock_runtime = bedrock_runtime_client()

set_page_config()
helpers.reset_session()

dataset = helpers.load_jsonl('data/titan_text.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6, 0.4])

with text:

    st.title('Amazon Titan Text')
    st.write("""Titan Text models are generative LLMs for tasks such as summarization, text generation (for example, creating a blog post), classification, open-ended Q&A, and information extraction. They are also trained on many different programming languages as well as rich text format like tables, JSON and csv’s among others.""")

    with st.expander("See Code"):
        st.code(helpers.render_titan_code('titan_text.jinja'), language="python")

    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:", height=st.session_state['height'], value=st.session_state["prompt"])
        submit = st.form_submit_button("Submit", type='primary')

    if prompt_data and submit:
        with st.spinner("Generating..."):

            response = helpers.invoke_model(bedrock_runtime, titan_generic(
                prompt_data), st.session_state['model'], max_tokens=st.session_state['max_tokens'], temperature=st.session_state['temperature'], top_p=st.session_state['top_p'])

            st.write("### Answer")
            st.info(response)

with code:
    helpers.tune_parameters('Amazon')

    st.subheader('Prompt Examples:')
    container2 = st.container(border=True)
    with container2:
        helpers.create_tabs(dataset)