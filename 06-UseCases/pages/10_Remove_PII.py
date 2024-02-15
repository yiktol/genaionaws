
import streamlit as st
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)

modelId = 'anthropic.claude-v2'
prompt = """Human: Here is some text. We want to remove all personally identifying information from this text and replace it with XXX. It's very important that names, phone numbers, and email addresses, gets replaced with XXX. 
Here is the text, inside <text></text> XML tags\n
<text>
   Joe: Hi Hannah!
   Hannah: Hi Joe! Are you coming over?  
   Joe: Yup! Hey I, uh, forgot where you live." 
   Hannah: No problem! It's 4085 Paco Ln, Los Altos CA 94306.
   Joe: Got it, thanks!  
</text> \n
Please put your sanitized version of the text with PII removed in <response></response> XML tags 

Assistant:
"""

with code:
    
    with st.form(key ='form2'):
        # provider = st.text_input('Provider', modelId.split('.')[0],disabled=True)
        # model_id=st.text_input('model_id',modelId,disabled=True)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1)
        top_p=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
        top_k=st.slider('topK', min_value = 0, max_value = 300, value = 250, step = 5)
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
    'top_k': {top_k},
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
    st.header("Remove PII")
    st.write("In this example, we want to remove all personally identifying information from the following text and replace it with XXX. It's very important that names, phone numbers, and email addresses are replaced with XXX.")
    with st.form('form1'):
        prompt = st.text_area(":orange[Prompt]", prompt, height=400)
        submit = st.form_submit_button("Remove PII",type='primary')
        
    if submit:
        output = invoke_model(client=bedrock_runtime_client(), prompt=prompt, model=modelId,
                             temperature=temperature,
                             top_p=top_p,
                             max_tokens=max_tokens_to_sample,
                             top_k=top_k,
                             )
        st.info(output)
    


