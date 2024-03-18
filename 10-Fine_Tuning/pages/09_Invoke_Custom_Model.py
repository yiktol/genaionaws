import json
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock = helpers.bedrock_client()
bedrock_runtime = helpers.bedrock_runtime_client()
region = 'us-east-1'


test_file_path = 'data/test-cnn-10.jsonl'
with open(test_file_path) as f:
    lines = f.read().splitlines()
    
test_prompt = json.loads(lines[3])['prompt']
reference_summary = json.loads(lines[3])['completion']

st.title("Invoking the Custom Model")

with st.expander(":orange[See Sample Test Prompt]"):
    st.subheader("Test Prompt")
    st.write(test_prompt)
    st.subheader("Reference Summary")
    st.write(reference_summary)


prompt = f"""{test_prompt}"""

accept = 'application/json'
contentType = 'application/json'

with st.container(border=True):
    prompt_data = st.text_area(":orange[Prompt]", value=prompt, height=300)
    
    col1, col2, col3 = st.columns([0.3,0.3,0.4])
    with col1:
        custom_model_id = st.selectbox("Custom Model",helpers.get_provisioned_models(),label_visibility="hidden")
        submit_custom = st.button("Submit to Fine Tuned Model", type="primary")
    with col2:
        base_model_id = st.selectbox("Base Model",helpers.get_models("Amazon"),label_visibility="hidden", index=6)
        submit_base = st.button("Submit to Base Model", type="primary")


base_model_arn = f'arn:aws:bedrock:us-east-1::foundation-model/{base_model_id}'

body = json.dumps(
    {
    "inputText": prompt_data,
    "textGenerationConfig": {
        "maxTokenCount": 2048,
        "stopSequences": ['User:'],
        "temperature": 0,
        "topP": 0.9
    }
    }
        )

if submit_custom:
    with st.spinner("Invoking the Custom Model..."):

        if helpers.get_provisioned_model_id() is None:
            st.write("Fine tuned model response: ")
            st.error("Please Submit the Provisioned Throughput Request")
        else:
            fine_tuned_response = bedrock_runtime.invoke_model(body=body, 
                                                    modelId=custom_model_id, 
                                                    accept=accept, 
                                                    contentType=contentType)
            fine_tuned_response_body = json.loads(fine_tuned_response.get('body').read())
            st.write("Fine tuned model response: ")
            st.success(fine_tuned_response_body["results"][0]["outputText"] + '\n')
    
if submit_base:
    with st.spinner("Invoking the Base Model..."):
        base_model_response = bedrock_runtime.invoke_model(body=body, 
                                                modelId=base_model_arn, 
                                                accept=accept, 
                                                contentType=contentType)
        base_model_response_body = json.loads(base_model_response.get('body').read())
        st.write("Base model response:")
        st.success(base_model_response_body["results"][0]["outputText"] + '\n')
