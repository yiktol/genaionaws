import streamlit as st
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.cohere as cohere

stlib.set_page_config()

suffix = 'cohere'
if suffix not in st.session_state:
    st.session_state[suffix] = {}


dataset = cohere.load_jsonl('data/cohere.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(cohere.params,suffix)

text, code = st.columns([0.6,0.4])

with text:
    st.title('Cohere')
    st.write('Cohere models are text generation models for business use cases. Cohere models are trained on data that supports reliable business applications, like text generation, summarization, copywriting, dialogue, extraction, and question answering.')

    with st.expander("See Code"): 
        st.code(cohere.render_cohere_code('command.jinja',suffix),language="python")
        
    with st.form("myform"):
        prompt_data = st.text_area("Enter your prompt here:",
            height = st.session_state[suffix]['height'],
            value = st.session_state[suffix]["prompt"]
        )
        submit = st.form_submit_button("Submit", type='primary')

    if prompt_data and submit:
        with st.spinner("Generating..."):
            generations = cohere.invoke_model(client=bedrock.runtime_client(), 
                                            prompt=prompt_data,
                                            model=st.session_state[suffix]['model'], 
                                            max_tokens  = st.session_state[suffix]['max_tokens'], 
                                            temperature = st.session_state[suffix]['temperature'], 
                                            p = st.session_state[suffix]['p'],
                                            k = st.session_state[suffix]['k'],
                                            stop_sequences = st.session_state[suffix]['stop_sequences'],
                                            stream = st.session_state[suffix]['stream'],
                                            num_generations = st.session_state[suffix]['num_generations'],
                                            return_likelihoods = st.session_state[suffix]['return_likelihoods']
                                            )

            st.write("### Answer")
            for index, generation in enumerate(generations):

                st.success(f"Generation {index + 1}\n")
                st.info(f"Text:\n {generation['text']}\n")
                if 'likelihood' in generation:
                    st.write(f"Likelihood:\n {generation['likelihood']}\n")
                
                st.info(f"Reason: {generation['finish_reason']}\n\n")


with code:

    cohere.tune_parameters('Cohere',suffix, index=1)


    st.subheader('Prompt Examples:')   
    with st.container(border=True) :
        stlib.create_tabs(dataset,suffix)
