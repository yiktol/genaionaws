import streamlit as st
import image.image_background_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Background")

if "image" not in st.session_state:
    st.session_state.image = "images/example.jpg"
if "mode" not in st.session_state:
    st.session_state.mode = 1
if "object" not in st.session_state:
    st.session_state.object = "Car"
if "prompt" not in st.session_state:
    st.session_state.prompt = "Car at the beach"
if "negative_prompt" not in st.session_state:
    st.session_state.negative_prompt = ""

col1, col2, col3 = st.columns(3)

options = [{"image": "images/flowers.png", "mode": "DEFAULT", "object": "Flowers", "prompt": "Flowers in the kitchen", "negative_prompt": ""},
           {"image": "images/dress.png","mode": "PRECISE", "object": "Woman in a long flowing dress","prompt": "Woman in a long flowing dress, in a city", "negative_prompt": ""},
           {"image": "images/robots.png", "mode": "DEFAULT", "object": "Robots", "prompt": "Robots in a bright neon robot arena", "negative_prompt": ""},
           {"image": "images/br.jpg", "mode": "DEFAULT", "object": "Bed", "prompt": "A stately bedroom", "negative_prompt": ""},
           {"image": "images/example.jpg", "mode": "PRECISE", "object": "Car", "prompt": "Car at the beach", "negative_prompt": ""},
           ]


def update_options(item_num):
    if options[item_num]["mode"] == "DEFAULT":
        st.session_state.mode = 0
    else:
        st.session_state.mode = 1
    st.session_state.object = options[item_num]["object"]
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.negative_prompt = options[item_num]["negative_prompt"]
    st.session_state.image = options[item_num]["image"]

def load_options(item_num):    
    st.write("Mode: ", options[item_num]["mode"])
    st.write("Object to keep:",options[item_num]["object"])
    st.write("Prompt:",options[item_num]["prompt"])
    st.write("Negative Prompt:",options[item_num]["negative_prompt"])
    st.button("Load Prompt", key=item_num+1, on_click=update_options, args=(item_num,))  


with col1:
    uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])
    
    if uploaded_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image(st.session_state.image)

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

with col2:
    st.subheader("Image parameters")
    
    with st.form("form1"):
        mask_prompt = st.text_input("Object to keep:", key="object", help="The mask text")
        
        prompt_text = st.text_area("Description including the object to keep and background to add:", key="prompt", height=100, help="The prompt text")
        
        negative_prompt = st.text_input("What should not be in the background:", help="The negative prompt", key="negative_prompt")

        outpainting_mode = st.radio("Outpainting mode:", ["DEFAULT", "PRECISE"], horizontal=True , index=st.session_state.mode)
        
        generate_button = st.form_submit_button("Generate", type="primary")


with col3:
    st.subheader("Result")

    if generate_button:
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
        else:
            image_bytes = glib.get_bytes_from_file(st.session_state.image)
        
        with st.spinner("Drawing..."):
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                image_bytes=image_bytes,
                mask_prompt=mask_prompt,
                negative_prompt=negative_prompt,
                outpainting_mode=outpainting_mode,
            )
        
        st.image(generated_image)
