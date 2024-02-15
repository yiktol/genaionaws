
import streamlit as st
import json
import boto3
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)

modelId = 'amazon.titan-text-lite-v1'
prompt = """Product: Sunglasses. 
Keywords: polarized, designer, comfortable, UV protection, aviators. 

Create a table that contains five variations of a detailed product description for the product listed above, each variation of the product description must use all the keywords listed.
"""

  
with code:
    
    with st.form(key ='form2'):
        # provider = st.text_input('Provider', modelId.split('.')[0],disabled=True)
        # model_id=st.text_input('model_id',modelId,disabled=True)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1)
        top_p=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
        max_tokens_to_sample=st.number_input('maxTokenCount',min_value = 50, max_value = 4096, value = 4096, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters')     
        
        code_data = f"""import json
import boto3

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )

modelId = 'amazon.titan-text-lite-v1'
accept = 'application/json'
contentType = 'application/json'

prompt= \"""{prompt}\"""

input = {{
        'inputText': prompt,
        'textGenerationConfig': {{
            'maxTokenCount': {max_tokens_to_sample},
            'stopSequences': [],
            'temperature': {temperature},
            'topP': {top_p}
        }}
    }}
body=json.dumps(input)
response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
response_body = json.loads(response.get('body').read())
results = response_body['results']
for result in results:
    print(result['outputText'])
"""        
        
    st.code(code_data, language="python")

with text:

    # st.title("Extract Action Items")
    st.header("Creating a Table of Product Descriptions")
    st.write("In this example, we want to create a table with five variations of a detailed product description for a specific product (sunglasses). Each variation of the product description must use all the keywords listed in the instruction.")
    with st.form('form1'):
        prompt = st.text_area(":orange[Product Descriptions]", prompt, height=200)
        submit = st.form_submit_button("Create Product Descriptions",type='primary')
        
    if submit:
        output = invoke_model(client=bedrock_runtime_client(), prompt=prompt, model=modelId,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens_to_sample)
        #print(output)
        st.write("Answer:")
        st.info(output)