import json
import streamlit as st
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.titan_text as titan_text

stlib.set_page_config()

suffix = 'streaming'
if suffix not in st.session_state:
    st.session_state[suffix] = {}
    
bedrock_runtime = bedrock.runtime_client()


dataset = titan_text.load_jsonl('data/streaming.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(titan_text.params,suffix)

text, code = st.columns([0.6,0.4])

with text:
    st.title('Bedrock Streaming')
    st.write("""Invoke the specified Amazon Bedrock model to run inference using the input provided. Return the response in a stream. To find out if a model supports streaming, call GetFoundationModel and check the responseStreamingSupported field in the response.""")


    with st.expander("See Code"):
        st.code(titan_text.render_titan_code('streaming.jinja',suffix),language="python")
        
    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height=st.session_state[suffix]['height'],
            value = st.session_state[suffix]["prompt"]  # Set default value
        )
        submit = st.form_submit_button("Submit", type='primary')

        model_id = st.session_state[suffix]['model']
        accept = 'application/json' 
        content_type = 'application/json'

        text_gen_config = {
            "maxTokenCount": st.session_state[suffix]['maxTokenCount'],
            "stopSequences": [], 
            "temperature": st.session_state[suffix]['temperature'],
            "topP": st.session_state[suffix]['topP']
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
            placeholder = st.empty()
            full_response = ''
            for event in response['body']:
                data = json.loads(event['chunk']['bytes'])
                chuck = data['outputText']
                full_response += chuck
                placeholder.info(full_response)
            placeholder.info(full_response)
                
            
with code:
    titan_text.tune_parameters('Amazon',suffix)

    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        stlib.create_tabs(dataset,suffix)

        
    