import streamlit as st
import utils.helpers as helpers
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.titan_text as titan_text


stlib.set_page_config()

suffix = 'titan_text'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

dataset = helpers.load_jsonl('data/titan_text.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(titan_text.params,suffix)

text, code = st.columns([0.6, 0.4])

with text:

    st.title('Amazon Titan Text')
    st.write("""Titan Text models are generative LLMs for tasks such as summarization, text generation (for example, creating a blog post), classification, open-ended Q&A, and information extraction. They are also trained on many different programming languages as well as rich text format like tables, JSON and csv’s among others.""")

    with st.expander("See Code"):
        st.code(titan_text.render_titan_code('titan_text.jinja',suffix), language="python")

    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:", height=st.session_state[suffix]['height'], value=st.session_state[suffix]["prompt"])
        submit = st.form_submit_button("Submit", type='primary')

    if prompt_data and submit:
        with st.spinner("Generating..."):

            response = titan_text.invoke_model(bedrock.runtime_client(), prompt_data, st.session_state[suffix]['model'], maxTokenCount=st.session_state[suffix]['maxTokenCount'], temperature=st.session_state[suffix]['temperature'], topP=st.session_state[suffix]['topP'])

            st.write("### Answer")
            st.info(response)

with code:
    titan_text.tune_parameters('Amazon',suffix)

    st.subheader('Prompt Examples:')
    container2 = st.container(border=True)
    with container2:
        stlib.create_tabs(dataset,suffix)