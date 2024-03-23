import utils.helpers as helpers
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.jurassic as jurassic
import streamlit as st

stlib.set_page_config()

suffix = 'jurassic'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

stlib.reset_session()

dataset = helpers.load_jsonl('data/jurassic.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(jurassic.params,suffix)

text, code = st.columns([0.6, 0.4])

bedrock_runtime = bedrock.runtime_client()

with text:

  st.title("AI21")
  st.write("AI21's Jurassic family of leading LLMs to build generative AI-driven applications and services leveraging existing organizational data. Jurassic supports cross-industry use cases including long and short-form text generation, contextual question answering, summarization, and classification. Designed to follow natural language instructions, Jurassic is trained on a massive corpus of web text and supports six languages in addition to English. ")

  with st.expander("See Code"):
      st.code(jurassic.render_jurassic_code('jurassic.jinja',suffix), language="python")

  with st.form("myform"):
    prompt_data = st.text_area(
        "Enter your prompt here:",
        height=st.session_state[suffix]['height'],
        value=st.session_state[suffix]["prompt"]
    )
    submit = st.form_submit_button("Submit", type='primary')

  if prompt_data and submit:
    with st.spinner("Generating..."):
      # Invoke the model
        response = jurassic.invoke_model(client=bedrock_runtime, 
                prompt=prompt_data, 
                model=st.session_state[suffix]['model'], 
                 maxTokens = st.session_state[suffix]['maxTokens'], 
                 temperature = st.session_state[suffix]['temperature'], 
                 topP = st.session_state[suffix]['topP'],
                 stop_sequences = [])

        st.write("### Answer")
        st.info(response)

with code:
  
    jurassic.tune_parameters('AI21',suffix, index=5)
    st.subheader('Prompt Examples:')
    with st.container(border=True):
        stlib.create_tabs(dataset,suffix)
