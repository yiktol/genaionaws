
import streamlit as st
import textwrap
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)

modelId = 'anthropic.claude-v2'
prompt = """Human: I'd like you to rewrite the following paragraph using the following instructions: "understandable to a 5th grader".\n 
"In 1758, the Swedish botanist and zoologist Carl Linnaeus published in his Systema Naturae, the two-word naming of species (binomial nomenclature). Canis is the Latin word meaning "dog", and under this genus, he listed the domestic dog, the wolf, and the golden jackal. "\n
Please put your rewrite in <rewrite></rewrite> tags.

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

prompt= \"{textwrap.shorten(prompt,width=50,placeholder='...')}\"

input = {{
    'prompt': prompt,
    'max_tokens_to_sample': {max_tokens_to_sample}, 
    'temperature': {temperature},
    'top_k': 250,
    'top_p': {top_p},
    'stop_sequences': []
}}

response = bedrock.invoke_model(
    body=json.dumps(input),
    modelId=modelId, 
    accept=accept,
    contentType=contentType
    )
    
response_body = json.loads(response.get('body').read())
completion = response_body['completion']
print(completion)
"""        
    
    st.code(code_data, language="python")




with text:

    # st.title("Extract Action Items")
    st.header("Content Generation")
    st.write("In this example, we want to rewrite the following paragraph with the explicit instruction that the generated content should be :orange[understandable to a 5th grader]. Also, we want the output to be in a specific formation.")
    with st.form('form1'):
        prompt = st.text_area(":orange[Prompt]", prompt, height=270)
        submit = st.form_submit_button("Generate Content",type='primary')
        
    if submit:
        output = invoke_model(client=bedrock_runtime_client(), prompt=prompt, model=modelId,
                             temperature=temperature,
                             top_p=top_p,
                             max_tokens=max_tokens_to_sample,
                             )
        st.write("Answer:")
        st.info(output)
    


