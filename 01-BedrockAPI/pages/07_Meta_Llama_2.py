import streamlit as st
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.llama as llama

stlib.set_page_config()

suffix = 'llama2'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

stlib.reset_session()

bedrock_runtime = bedrock.runtime_client()

dataset = llama.load_jsonl('data/meta.jsonl')

stlib.initsessionkeys(dataset[0], suffix)
stlib.initsessionkeys(llama.params, suffix)

text, code = st.columns([0.6, 0.4])

with text:

    st.title("Meta")
    st.write("Llama is a family of large language models that uses publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.")

    with st.expander("See Code"):
        st.code(llama.render_meta_code(
            'llama.jinja', suffix), language="python")

    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area("Enter your prompt here:",
                                   height=st.session_state[suffix]['height'],
                                   value=st.session_state[suffix]["prompt"]
                                   )
        submit = st.form_submit_button("Submit", type='primary')

    if prompt_data and submit:
        with st.spinner("Generating..."):
            response = llama.invoke_model(bedrock_runtime,
                                          prompt_data,
                                          st.session_state[suffix]['model'],
                                          max_gen_len=st.session_state[suffix]['max_gen_len'],
                                          temperature=st.session_state[suffix]['temperature'],
                                          top_p=st.session_state[suffix]['top_p']
                                          )

            st.write("### Answer")
            st.info(response)


with code:

    llama.tune_parameters('Meta', suffix, index=3)

    st.subheader('Prompt Examples:')
    with st.container(border=True):
        stlib.create_tabs(dataset, suffix)
