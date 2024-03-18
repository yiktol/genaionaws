
import streamlit as st
import textwrap
from helpers import bedrock_runtime_client, set_page_config, invoke_model, getmodelId


set_page_config()


text, code = st.columns([0.6,0.4])

modelId = 'ai21.j2-mid'
prompt = """Write an engaging product description for a clothing eCommerce site. Make sure to include the following features in the description.\n
Product: Humor Men's Graphic T-Shirt.\n
Features: 
- Soft cotton 
- Short sleeve
- Have a print of Einstein's quote: "artificial intelligence is no match for natural stupidity” 

Description: 
"""


    
with code:
    with st.container(border=True):
        provider = st.selectbox('Provider',('Amazon','Anthropic','AI21','Cohere','Meta'))
        model_id=st.text_input('model_id',getmodelId(provider))
        
    with st.form(key ='form2'):
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
        topP=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
        maxTokens=st.number_input('maxTokens',min_value = 50, max_value = 4096, value = 1024, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters')    

    code_data = f"""import json
import boto3

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )

modelId = 'ai21.j2-mid'
accept = 'application/json'
contentType = 'application/json'

prompt= \"{textwrap.shorten(prompt,width=50,placeholder='...')}\"

input = {{
    'prompt':prompt, 
    'maxTokens': {maxTokens},
    'temperature': {temperature},
    'topP': {topP},
    'stopSequences': [],
    'countPenalty': {{'scale': 0}},
    'presencePenalty': {{'scale': 0}},
    'frequencyPenalty': {{'scale': 0}}
        }}
        
response = bedrock.invoke_model(
    body=json.dumps(input),
    modelId=modelId, 
    accept=accept,
    contentType=contentType
    )
    
response_body = json.loads(response.get('body').read())
completions = response_body['completions']

for part in completions:
    print(part['data']['text'])

"""

    with st.expander("Show Code"):
        st.code(code_data, language="python")

with text:

    # st.title("Extract Action Items")
    st.header("Generate Product Descriptions")
    st.markdown("""In this example, we want to write an engaging product description for a clothing eCommerce site. We want to include the following features in the description:
- Soft cotton
- Short sleeve
- Have a print of Einstein's quote: \"artificial intelligence is no match for natural stupidity\"""")

    with st.form('form1'):
        prompt = st.text_area(":orange[Product Descriptions]", prompt, height=300)
        submit = st.form_submit_button("Generate Product Descriptions",type='primary')
        
    if submit:
        with st.spinner("Thinking..."):
            output = invoke_model(client=bedrock_runtime_client(), 
                                prompt=prompt, 
                                model=model_id,
                                    temperature=temperature,
                                top_p=topP,
                                max_tokens=maxTokens,)
            #print(output)
            st.write("Answer:")
            st.info(output)