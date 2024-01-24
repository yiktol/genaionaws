import streamlit as st
import boto3
import json
from helpers import set_page_config

set_page_config()

#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

with st.sidebar:
    with st.form(key ='Form1'):
        "Parameters:"
        temperature =st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_k=st.number_input('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        submitted1 = st.form_submit_button(label = 'Set Parameters')
        
    
st.title("📝 File Q&A with Anthropic")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)
submit = st.button("Submit")

if uploaded_file and question:
    article = uploaded_file.read().decode()
    prompt = f"""Human: Here's an article:\n\n<article>
    {article}\n\n</article>\n\n{question} \n\nAssistant:"""


    body = {"prompt": prompt,
            "max_tokens_to_sample": 300, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31"}

    body = json.dumps(body) # Encode body as JSON string

    modelId = 'anthropic.claude-instant-v1' 
    accept = 'application/json'
    contentType = 'application/json'

    #Invoke the model
    response = bedrock_runtime.invoke_model(body=body.encode('utf-8'), # Encode to bytes
                                    modelId=modelId, 
                                    accept=accept, 
                                    contentType=contentType)
    
    response_body = json.loads(response.get('body').read())
    print(response_body.get('completion'))
    st.write("### Answer")
    st.write(response_body.get('completion'))