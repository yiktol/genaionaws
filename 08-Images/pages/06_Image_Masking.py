import streamlit as st
import image.image_masking_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Masking")


col1, col2, col3 = st.columns(3)

if "masking_mode" not in st.session_state:
    st.session_state.masking_mode = 0
if "painting_mode" not in st.session_state:
    st.session_state.painting_mode = 0
if "mask_prompt" not in st.session_state:
    st.session_state.mask_prompt = ""
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = "Framed photographs over a desk, with a stool under the desk"

options = [{"masking_mode": 0, "painting_mode": 0, "mask_prompt": "", "prompt_text": "Framed photographs over a desk, with a stool under the desk"},
           {"masking_mode": 0, "painting_mode": 1, "mask_prompt": "", "prompt_text": "A cozy living room"},
           {"masking_mode": 1, "painting_mode": 0, "mask_prompt": "painting", "prompt_text": "embedded fireplace"},
           {"masking_mode": 1, "painting_mode": 1, "mask_prompt": "painting", "prompt_text": "living room"}]
    
def load_options(item_num):
    if options[item_num]["masking_mode"] == 0:
        st.write("Masking Mode: Image")
    else:
        st.write("Masking Mode: Prompt")
    st.write("Mask Prompt:", options[item_num]["mask_prompt"])
    if options[item_num]["painting_mode"] == 0:
        st.write("Painting Mode: INPAINTING")
    else:
        st.write("Painting Mode: OUTPAINTING")
    st.write("Prompt:", options[item_num]["prompt_text"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(item_num,))

    
def update_options(item_num):
    st.session_state.masking_mode = options[item_num]["masking_mode"]
    st.session_state.painting_mode = options[item_num]["painting_mode"]
    st.session_state.mask_prompt = options[item_num]["mask_prompt"]
    st.session_state.prompt_text = options[item_num]["prompt_text"]


with col1:
    st.subheader("Image")
    container1 = st.container(border=True)
    with container1:
        uploaded_image_file = st.file_uploader("Select an image", type=['png', 'jpg'])
        
        if uploaded_image_file:
            uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_image_file.getvalue())
            st.image(uploaded_image_preview)
        else:
            st.image("images/desk1.jpg")
            
    container2 = st.container(border=True)    
    with container2:
        st.subheader('Prompt Examples:')
        tab1, tab2, tab3, tab4 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4"])
        with tab1:
            load_options(item_num=0)  
        with tab2:
            load_options(item_num=1)  
        with tab3:
            load_options(item_num=2) 
        with tab4:
            load_options(item_num=3) 
            
with col2:
    st.subheader("Mask")
    
    container3 = st.container(border=True)    
    with container3:    
        masking_mode = st.radio("Masking mode:", ["Image", "Prompt"], index = st.session_state.masking_mode ,  horizontal=True)
        
        if masking_mode == 'Image':
        
            uploaded_mask_file = st.file_uploader("Select an image mask", type=['png', 'jpg'])
            
            if uploaded_mask_file:
                uploaded_mask_preview = glib.get_bytesio_from_bytes(uploaded_mask_file.getvalue())
                st.image(uploaded_mask_preview)
            else:
                st.image("samples/mask1.png")
        else:
            mask_prompt = st.text_input("Item to mask:", key="mask_prompt", help="The item to replace (if inpainting), or keep (if outpainting).")
        
        
with col3:
    st.subheader("Result")

    container4 = st.container(border=True)    
    with container4:   
    
        prompt_text = st.text_area("Prompt text:", height=100, key="prompt_text" ,help="The prompt text")

        painting_mode = st.radio("Painting mode:", ["INPAINTING", "OUTPAINTING"], index=st.session_state.painting_mode)
        
        generate_button = st.button("Generate", type="primary")

        if generate_button:
            with st.spinner("Drawing..."):
                
                if uploaded_image_file:
                    image_bytes = uploaded_image_file.getvalue()
                else:
                    image_bytes = glib.get_bytes_from_file("images/desk1.jpg")
                
                if masking_mode == 'Image':
                    if uploaded_mask_file:
                        mask_bytes = uploaded_mask_file.getvalue()
                    else:
                        mask_bytes = glib.get_bytes_from_file("samples/mask1.png")
                    
                    generated_image = glib.get_image_from_model(
                        prompt_content=prompt_text, 
                        image_bytes=image_bytes,
                        masking_mode=masking_mode,
                        mask_bytes=mask_bytes,
                        painting_mode=painting_mode
                    )
                else:
                    generated_image = glib.get_image_from_model(
                        prompt_content=prompt_text, 
                        image_bytes=image_bytes,
                        masking_mode=masking_mode,
                        mask_prompt=mask_prompt,
                        painting_mode=painting_mode
                    )
                
            
            st.image(generated_image)
