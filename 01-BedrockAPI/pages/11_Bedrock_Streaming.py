import json
import streamlit as st
from utils import set_page_config, bedrock_runtime_client, titan_generic
import utils.helpers as helpers

set_page_config()

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

helpers.reset_session()

dataset = helpers.load_jsonl('data/streaming.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6,0.4])

with text:
    st.title('Bedrock Streaming')
    st.write("""Invoke the specified Amazon Bedrock model to run inference using the input provided. Return the response in a stream. To find out if a model supports streaming, call GetFoundationModel and check the responseStreamingSupported field in the response.""")


    with st.expander("See Code"):
        st.code(helpers.render_titan_code('streaming.jinja'),language="python")
        
    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height=st.session_state['height'],
            value = titan_generic(st.session_state["prompt"])  # Set default value
        )
        submit = st.form_submit_button("Submit", type='primary')

        model_id = st.session_state['model']
        accept = 'application/json' 
        content_type = 'application/json'

        text_gen_config = {
            "maxTokenCount": st.session_state['max_tokens'],
            "stopSequences": [], 
            "temperature": st.session_state['temperature'],
            "topP": st.session_state['top_p']
            }

        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig": text_gen_config  
        })


    if submit:
        with st.spinner("Streaming..."):
        #invoke the model with a streamed response 
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body, 
                modelId=model_id, 
                accept=accept, 
                contentType=content_type
            )

            st.write("### Answer")
            for event in response['body']:
                data = json.loads(event['chunk']['bytes'])
                #print(data['outputText'])
                st.info(data['outputText'])
            
with code:
    helpers.tune_parameters('Amazon')

    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        helpers.create_tabs(dataset)

        
    