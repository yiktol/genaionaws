
import streamlit as st
import textwrap
from helpers import bedrock_runtime_client, set_page_config, invoke_model,getmodelId


set_page_config()


text, code = st.columns([0.6,0.4])


modelId = 'amazon.titan-tg1-large'
prompt = """Miguel: Hi Brant, I want to discuss the workstream  for our new product launch\n
Brant: Sure Miguel, is there anything in particular you want to discuss?\n
Miguel: Yes, I want to talk about how users enter into the product.\n
Brant: Ok, in that case let me add in Namita.\n
Namita: Hey everyone\n
Brant: Hi Namita, Miguel wants to discuss how users enter into the product.\n
Miguel: its too complicated and we should remove friction.  for example, why do I need to fill out additional forms?  I also find it difficult to find where to access the product when I first land on the landing page.\n
Brant: I would also add that I think there are too many steps.\n 
Namita: Ok, I can work on the landing page to make the product more discoverable but brant can you work on the additonal forms?\n
Brant: Yes but I would need to work with James from another team as he needs to unblock the sign up workflow.  Miguel can you document any other concerns so that I can discuss with James only once?\n
Miguel: Sure.\n
From the meeting transcript above, Create a list of action items for each person. 
"""


with code:
    with st.container(border=True):
        provider = st.selectbox('Provider',('Amazon','Anthropic','AI21','Cohere','Meta'))
        model_id=st.text_input('model_id',getmodelId(provider))

    with st.form(key ='form2'):
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1)
        top_p=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
        max_tokens_to_sample=st.number_input('maxTokenCount',min_value = 50, max_value = 4096, value = 2048, step = 1)
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

prompt= \"{textwrap.shorten(prompt,width=50,placeholder='...')}\"

input = {{
        'inputText': prompt,
        'textGenerationConfig': {{
            'maxTokenCount': {max_tokens_to_sample},
            'stopSequences': [],
            'temperature': {temperature},
            'topP': {top_p}
        }}
    }}
    
response = bedrock.invoke_model(
    body=json.dumps(input),
    modelId=modelId, 
    accept=accept,
    contentType=contentType
    )
    
response_body = json.loads(response.get('body').read())
results = response_body['results']

for result in results:
    print(result['outputText'])
        """

    with st.expander("Show Code"):
        st.code(code_data, language="python")


with text:

    # st.title("Extract Action Items")
    st.header("Action Items from a Meeting Transcript")
    st.write("In this example, we want to extract action items from the following meeting transcript:")
    
    with st.form('form1'):
        prompt = st.text_area(":orange[Meeting Transcript]", prompt, height=670)
        submit = st.form_submit_button("Extract Action Items",type='primary')
        
    if submit:
        with st.spinner("Thinking..."):
            output = invoke_model(
                client=bedrock_runtime_client(), 
                prompt=prompt, 
                model=model_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens_to_sample,
                )
            #print(output)
            st.write("Answer:")
            st.info(output)
        

