import textwrap
import streamlit as st
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)


modelId = 'amazon.titan-text-express-v1'
prompt = """Bob has been at the company for 3.5 years and is up for promotion for senior manager.\n  
Bob has the following accomplishments: 
- led the delivery of a new software product for productivity. 
- mentored 10 employees. 
- presented to 20 different end customers  which converted into 18 new users. 

Using  the above content, write a business professional narrrative about Bobs performance.\n
The narrative must include the reasons why bob should be promoted and mention three amazons leadership principles. 
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

modelId = 'amazon.titan-text-express-v1'
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

    st.code(code_data, language="python")


with text:

    # st.title("Extract Action Items")
    st.header("Write a Promo Doc")
    st.markdown("""In this example, we want to write a promo doc for Bob using the following content. The writing must include the reasons why bob should be promoted and mention three amazons leadership principles.\n
Bob has been at the company for 3.5 years and is up for promotion for senior manager
Bob has the following accomplishments:
- led the delivery of a new software product for productivity.
- mentored 10 employees.
- presented to 20 different end customers which converted into 18 new users.""")
    
    with st.form('form1'):
        prompt = st.text_area(":orange[Prompt:]", prompt, height=350)
        submit = st.form_submit_button("Write a Promo Doc",type='primary')
        
    if submit:
        output = invoke_model(
            client=bedrock_runtime_client(), 
            prompt=prompt, 
            model=modelId,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens_to_sample,
            )
        #print(output)
        st.write("Answer:")
        st.info(output)
        

