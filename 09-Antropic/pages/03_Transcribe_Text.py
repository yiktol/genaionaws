import boto3
import json
import streamlit as st
from helpers import set_page_config, bedrock_runtime_client, get_base64_encoded_image
from IPython.display import Image
import base64

set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

col1, col2 = st.columns(2)

if "image" not in st.session_state:
    st.session_state.image = "images/transcribe/stack_overflow.png"
if "prompt" not in st.session_state:
    st.session_state.prompt = "Transcribe the code in the answer. Only output the code."
if "media_type" not in st.session_state:
    st.session_state.media_type = "image/png"

col1, col2 = st.columns(2)

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.text_input('model', 'anthropic.claude-3-sonnet-20240229-v1:0', disabled=True)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_k=st.slider('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        top_p=st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens',min_value = 50, max_value = 4096, value = 2048, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters') 

    st.sidebar.button(label='Clear Image Cache', on_click=form_callback)        
#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

modelId = model
accept = 'application/json'
contentType = 'application/json'

Image(filename=st.session_state.image) 


options = [{"image": "images/transcribe/stack_overflow.png", "media_type": "image/png","prompt": "Transcribe the code in the answer. Only output the code."},
           {"image": "images/transcribe/school_notes.png", "media_type": "image/png","prompt": "Transcribe this text. Only output the text and nothing else."},
            {"image": "images/transcribe/vehicle_form.jpg", "media_type": "image/jpeg","prompt": "Transcribe this form exactly."},
            {"image": "images/transcribe/page.jpeg", "media_type": "image/jpeg","prompt": "Which is the most critical issue for live rep support?"},
            {"image": "images/transcribe/org_chart.jpeg", "media_type": "image/jpeg","prompt": "Turn this org chart into JSON indicating who reports to who. Only output the JSON and nothing else."},
 ]

def update_options(item_num):
    st.session_state.media_type = options[item_num]["media_type"]  # "image/jpeg" or "image/png"
    st.session_state.image = options[item_num]["image"]  # "images/best_practices/nine_dogs.jpg"
    st.session_state.prompt = options[item_num]["prompt"]  # "How many dogs are in this picture?"


def load_options(item_num):    
    st.write("Image:",options[item_num]["image"] )
    st.write("Media Type:",options[item_num]["media_type"]) 
    st.write("Prompt:",options[item_num]["prompt"])
    st.button("Load Prompt", key=item_num+1, on_click=update_options, args=(item_num,))  


with col2: 
    st.subheader("Image")
    st.image(st.session_state.image)
  

with col1:
    st.subheader("Prompt")
    with st.form("myform"):
        prompt_data = st.text_area(
            "Enter your prompt here:",
            height = 100,
            key="prompt")
        submit = st.form_submit_button("Submit", type="primary")
    
  
    container2 = st.container(border=True)    
    with container2:
        st.subheader('Prompt Examples:')
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"])
        with tab1:
            load_options(item_num=0)
        with tab2:
            load_options(item_num=1)
        with tab3:
            load_options(item_num=2)
        with tab4:
            load_options(item_num=3)
        with tab5:
            load_options(item_num=4)

    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": st.session_state.media_type, "data": get_base64_encoded_image(st.session_state.image)}},
                {"type": "text", "text": st.session_state.prompt}
                ]
        }
    ]  

    body = {"max_tokens": max_tokens_to_sample, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
            "messages": message_list}

            
    if prompt_data and submit:
        #Invoke the model
        with st.spinner("Thinking..."):
            response = bedrock_runtime.invoke_model(body=json.dumps(body), # Encode to bytes
                                            modelId=modelId, 
                                            accept=accept, 
                                            contentType=contentType)

            response_body = json.loads(response.get('body').read())
            # print(response_body.get('content'))
            
            col2.write("### Answer")
            col2.info(response_body.get('content')[0]['text'])