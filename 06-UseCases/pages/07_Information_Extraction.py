
import streamlit as st
import json
import boto3
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)

modelId = 'anthropic.claude-v2'
prompt = """Human: Please precisely copy any email addresses from the following text and then write them in a table with index number.. Only write an email address if it's precisely spelled out in the input text. If there are no email addresses in the text, write "N/A". Do not say anything else.\n
"Phone Directory:
John Latrabe, 800-232-1995, john909709@geemail.com
Josie Lana, 800-759-2905, josie@josielananier.com
Keven Stevens, 800-980-7000, drkevin22@geemail.com 
Phone directory will be kept up to date by the HR manager." 

Assistant:
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

modelId = 'anthropic.claude-v2'
accept = 'application/json'
contentType = 'application/json'

prompt= \"""{prompt}\"""

input = {{
    'prompt': prompt,
    'max_tokens_to_sample': {max_tokens_to_sample}, 
    'temperature': {temperature},
    'top_k': 250,
    'top_p': {top_p},
    'stop_sequences': []
}}
body=json.dumps(input)
response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
response_body = json.loads(response.get('body').read())
completion = response_body['completion']
print(completion)
"""
    
    st.code(code_data, language="python")



with text:

    # st.title("Extract Action Items")
    st.header("Information Extraction")
    st.write("In this example, we want to extract email addresses from a phone directory. More specifically, we want to write the email address only when it is precisely spelled out in the text. If there are no email addresses in the text, write \"N/A\".")
    with st.form('form1'):
        prompt = st.text_area(":orange[Prompt]", prompt, height=300)
        submit = st.form_submit_button("Extract",type='primary')
        
    if submit:
        output = invoke_model(client=bedrock_runtime_client(), prompt=prompt, model=modelId,
                             temperature=temperature,
                             top_p=top_p,
                             max_tokens=max_tokens_to_sample,
                             )
        #print(output)
        st.write("Answer:")
        st.info(output)
    


