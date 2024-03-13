import json
from utils import set_page_config, bedrock_runtime_client, mistral_generic
import utils.helpers as helpers
import streamlit as st

set_page_config()

helpers.reset_session()

bedrock_runtime = bedrock_runtime_client(region='us-west-2')

dataset = helpers.load_jsonl('data/mistral.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6,0.4])

with text:
    st.title('Mistral AI')
    st.markdown("""Mistral AI is a small creative team with high scientific standards. We make efficient, helpful and trustworthy AI models through ground-breaking innovations.
- A 7B dense Transformer, fast-deployed and easily customisable. Small, yet powerful for a variety of use cases. Supports English and code, and a 32k context window.
- A 7B sparse Mixture-of-Experts model with stronger capabilities than Mistral 7B. Uses 12B active parameters out of 45B total. Supports multiple languages, code and 32k context window.
                """)


    with st.expander("See Code"):
        st.code(helpers.render_mistral_code('instruct.jinja'),language="python")

    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = st.session_state['height'],
            value = st.session_state["prompt"]
        )
        submit = st.form_submit_button("Submit", type='primary')

    model_id = st.session_state['model']
    accept = 'application/json' 
    content_type = 'application/json'

    body = json.dumps({
        "prompt": mistral_generic(prompt_data),
        "max_tokens": st.session_state['max_tokens'],
        "temperature": st.session_state['temperature'],
        "top_p": st.session_state['top_p']
    })

    if prompt_data and submit:
        with st.spinner("Generating..."):
            response = bedrock_runtime.invoke_model(body=body.encode('utf-8'), # Encode to bytes
                                            modelId=model_id, 
                                            accept=accept, 
                                            contentType=content_type)

            response_body = json.loads(response.get('body').read().decode('utf-8'))

            st.write("Answer")
            st.info(f"{response_body.get('outputs')[0].get('text')}")

with code:
    helpers.tune_parameters('Mistral', index=0,region='us-west-2')
    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        helpers.create_tabs(dataset)



