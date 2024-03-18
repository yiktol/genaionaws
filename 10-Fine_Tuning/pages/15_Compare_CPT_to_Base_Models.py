import json
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock = helpers.bedrock_client()
bedrock_runtime = helpers.bedrock_runtime_client()
region = 'us-east-1'


st.title("Invoking the Continued Pre-training Model")
st.markdown("Provision the customized model and compare the answer against the base model to evaluate the improvement")

prompt = """Write aws-cli bash script to create a dynamoDB table. 
Do not repeat answer.
Do not add any preamble. 
"""

base_model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1'

accept = 'application/json'
contentType = 'application/json'

with st.container(border=True):
    prompt_data = st.text_area(":orange[Prompt:]", value=prompt, height=100)
    
    col1, col2, col3 = st.columns([0.3,0.3,0.4])
    with col1:
        custom_model_id = st.selectbox(":orange[Custom Model]",helpers.get_provisioned_models(),label_visibility="visible")
        submit = st.button("Compare", type="primary")
    with col2:
        base_model_id = st.selectbox(":orange[Base Model]",helpers.get_models("Amazon"),label_visibility="visible", index=6)

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

if submit:
    with st.spinner("Invoking the Custom Model..."):
        
        provisioned_model_id = helpers.get_provisioned_model_id()
        
        if provisioned_model_id is None:
            st.write("Continued Pre-training model response: ")
            st.error("Please Submit the Provisioned Throughput Request")
        else:
            fine_tuned_response = bedrock_runtime.invoke_model(body=body, 
                                                    modelId=custom_model_id, 
                                                    accept=accept, 
                                                    contentType=contentType)
            fine_tuned_response_body = json.loads(fine_tuned_response.get('body').read())
            st.write("Continued Pre-training model response: ")
            st.info(fine_tuned_response_body["results"][0]["outputText"] + '\n')
    
    with st.spinner("Invoking the Base Model..."):
        
        base_model_response = bedrock_runtime.invoke_model(body=body, 
                                                modelId=base_model_arn, 
                                                accept=accept, 
                                                contentType=contentType)
        base_model_response_body = json.loads(base_model_response.get('body').read())
        st.write("Base model response:")
        st.success(base_model_response_body["results"][0]["outputText"] + '\n')
