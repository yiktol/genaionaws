import boto3
import json
import streamlit as st
from helpers import set_page_config, bedrock_runtime_client, get_base64_encoded_image
from IPython.display import Image
import base64

set_page_config()

col1, col2 = st.columns(2)

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

if "message_count" not in st.session_state:
    st.session_state.message_count = 0
if "image" not in st.session_state:
    st.session_state.image = "images/best_practices/nine_dogs.jpg"
if "prompt" not in st.session_state:
    st.session_state.prompt = "How many dogs are in this picture?"
if "media_type" not in st.session_state:
    st.session_state.media_type = "image/jpeg"
if "message_list" not in st.session_state:
    st.session_state.message_list = [
    {
        "role": 'user',
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": st.session_state.media_type, "data": get_base64_encoded_image(st.session_state.image)}},
            {"type": "text", "text": st.session_state.prompt}
            ]
    }
]



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


options = [{"image": "images/best_practices/nine_dogs.jpg", "media_type": "image/jpeg","prompt": "How many dogs are in this picture?"},
           {"image": "images/best_practices/nine_dogs.jpg", "media_type": "image/jpeg","prompt": "You have perfect vision and pay great attention to detail which makes you an expert at counting objects in images. How many dogs are in this picture? Before providing the answer in <answer> tags, think step by step in <thinking> tags and analyze every part of the image."},
           {"image": "images/best_practices/circle.png", "media_type": "image/png","prompt": "Describe the image."},
           {"image": "images/best_practices/labeled_circle.png", "media_type": "image/png","prompt": "Calculate"},
           {"image": "images/best_practices/table.png", "media_type": "image/png","prompt": "What's the difference between these two numbers?"},
           {"image": "images/best_practices/140.png", "media_type": "image/png","prompt": "What speed am I going?"},
           {"image": "images/best_practices/140.png", "media_type": "image/png","prompt": "What speed am I going?"},
           {"image": "images/best_practices/receipt1.png", "media_type": "image/png","prompt": "Output the name of the restaurant and the total."},
           {"image": "images/best_practices/officer_example.png", "media_type": "image/png","prompt": "These pants are (in order) WRINKLE-RESISTANT DRESS PANT, ITALIAN MELTON OFFICER PANT, SLIM RAPID MOVEMENT CHINO. What pant is shown in the last image?"}
           ]

def load_options(item_num):   
    st.write("Image:",options[item_num]["image"] )
    st.write("Media Type:",options[item_num]["media_type"]) 
    st.write("Prompt:",options[item_num]["prompt"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(item_num,))
    
def update_options(item_num):
    st.session_state.media_type = options[item_num]["media_type"]  # "image/jpeg" or "image/png"
    st.session_state.image = options[item_num]["image"]  # "images/best_practices/nine_dogs.jpg"
    st.session_state.prompt = options[item_num]["prompt"]  # "How many dogs are in this picture?"
    st.session_state.message_count = item_num  # message_list

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
    
 
    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        # st.subheader('Prompt Examples:')
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6", "Prompt7", "Prompt8", "Prompt9"])
        with tab1:
            load_options(0)
        with tab2:
            load_options(1)
        with tab3:
            load_options(2)
        with tab4:
            load_options(3)
        with tab5:
            load_options(4)
        with tab6:
            load_options(5)
        with tab7:
            load_options(6)
        with tab8:
            load_options(7)
        with tab9:
            load_options(8)

    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": st.session_state.media_type, "data": get_base64_encoded_image(st.session_state.image)}},
                {"type": "text", "text": prompt_data}
                ]
        }
    ]   
    
    message_list_8 = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image("images/best_practices/receipt1.png")}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image("images/best_practices/receipt2.png")}},
                {"type": "text", "text": "Output the name of the restaurant and the total."}
            ]
        }
    ]

    message_list_9 = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image("images/best_practices/wrinkle.png")}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image("images/best_practices/officer.png")}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image("images/best_practices/chinos.png")}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image("images/best_practices/officer_example.png")}},
                {"type": "text", "text": "These pants are (in order) WRINKLE-RESISTANT DRESS PANT, ITALIAN MELTON OFFICER PANT, SLIM RAPID MOVEMENT CHINO. What pant is shown in the last image?"}
            ]
        }
    ]


    message_list_7 = [
    {
        "role": 'user',
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image("images/best_practices/70.png")}},
            {"type": "text", "text": prompt_data}
        ]
    },
    {
        "role": 'assistant',
        "content": [
            {"type": "text", "text": "You are going 70 miles per hour."}
        ]
    },
    {
        "role": 'user',
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": st.session_state.media_type, "data": get_base64_encoded_image("images/best_practices/100.png")}},
            {"type": "text", "text": prompt_data}
        ]
    },
    {
        "role": 'assistant',
        "content": [
            {"type": "text", "text": "You are going 100 miles per hour."}
        ]
    },
    {
        "role": 'user',
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": st.session_state.media_type, "data": get_base64_encoded_image("images/best_practices/140.png")}},
            {"type": "text", "text": prompt_data}
        ]
    }
    ]     
    
    
    if st.session_state.message_count == 6:
        messagelist = message_list_7
        col1.image("images/best_practices/70.png")
        col1.write("Assistant: You are going 70 miles per hour.")
        col1.image("images/best_practices/100.png")
        col1.write("Assistant: You are going 100 miles per hour.")
    elif st.session_state.message_count == 7:
        messagelist = message_list_8
        col2.image("images/best_practices/receipt2.png")
    elif st.session_state.message_count == 8:
        messagelist = message_list_9
        col1.image("images/best_practices/wrinkle.png")
        col1.image("images/best_practices/officer.png")
        col1.image("images/best_practices/chinos.png")
    else:
        messagelist = message_list
    
    body = {"max_tokens": max_tokens_to_sample, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messagelist}     
     
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