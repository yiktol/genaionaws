import streamlit as st
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.claude2 as claude2

stlib.set_page_config()

suffix = 'claude2'
if suffix not in st.session_state:
    st.session_state[suffix] = {}


dataset = claude2.load_jsonl('data/anthropic.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(claude2.params,suffix)

text, code = st.columns([0.6,0.4])


with text:
    st.title("Anthropic Claude 2")
    st.write("""Anthropic offers the Claude family of large language models purpose built for conversations, 
            summarization, Q&A, workflow automation, coding and more. 
            Early customers report that Claude is much less likely to produce harmful outputs, 
            easier to converse with, and more steerable - so you can get your desired output with less effort. 
            Claude can also take direction on personality, tone, and behavior.""")

    with st.expander("See Code"): 
        st.code(claude2.render_claude_code('claude.jinja',suffix),language="python")

    # Define prompt and model parameters
    prompt_input = "Write a python code that list all countries."
    
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = st.session_state[suffix]['height'],
            value = st.session_state[suffix]["prompt"]
        )
        submit = st.form_submit_button("Submit", type='primary')
        
    if prompt_data and submit:
        with st.spinner("Generating..."):
            response = claude2.invoke_model(client=bedrock.runtime_client(), 
                                            prompt=prompt_data,
                                            model=st.session_state[suffix]['model'], 
                                            max_tokens_to_sample  = st.session_state[suffix]['max_tokens_to_sample'], 
                                            temperature = st.session_state[suffix]['temperature'], 
                                            top_p = st.session_state[suffix]['top_p'],
                                            top_k = st.session_state[suffix]['top_k'])

            st.write("### Answer")
            st.info(response)

with code:

    claude2.tune_parameters('Anthropic',suffix,index=6)
    st.subheader('Prompt Examples:')   
    with st.container(border=True):
        stlib.create_tabs(dataset,suffix)