
import streamlit as st
import json
import boto3
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)

modelId = 'ai21.j2-mid'
prompt = """Write an engaging product description for a clothing eCommerce site. Make sure to include the following features in the description. 
Product: Humor Men's Graphic T-Shirt.\n
Features: 
- Soft cotton 
- Short sleeve
- Have a print of Einstein's quote: "artificial intelligence is no match for natural stupidity” 

Description: 
"""

code_data = f"""import json
import boto3

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )

modelId = 'ai21.j2-mid'
accept = 'application/json'
contentType = 'application/json'

prompt= \"""{prompt}\"""

input = {{
    'prompt':prompt, 
    'maxTokens': 200,
    'temperature': 0.3,
    'topP': 1.0,
    'stopSequences': [],
    'countPenalty': {{'scale': 0}},
    'presencePenalty': {{'scale': 0}},
    'frequencyPenalty': {{'scale': 0}}
        }}
body=json.dumps(input)
response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
response_body = json.loads(response.get('body').read())
completions = response_body['completions']
for part in completions:
    print(part['data']['text'])

"""

with text:

    # st.title("Extract Action Items")
    st.header("Generate Product Descriptions")
    st.markdown("""In this example, we want to write an engaging product description for a clothing eCommerce site. We want to include the following features in the description:

- Soft cotton
- Short sleeve
- Have a print of Einstein's quote: \"artificial intelligence is no match for natural stupidity\")""")

    with st.form('form1'):
        prompt = st.text_area(":orange[Product Descriptions]", prompt, height=300)
        submit = st.form_submit_button("Generate Product Descriptions",type='primary')
        
    if submit:
        output = invoke_model(client=bedrock_runtime_client(), prompt=prompt, model=modelId)
        #print(output)
        st.info(output)
    
with code:
    

    st.code(code_data, language="python")

