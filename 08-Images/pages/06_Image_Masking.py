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
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = "Framed photographs over a desk, with a stool under the desk"

def on_1_click():
    st.session_state.masking_mode = 0
    st.session_state.painting_mode = 0
    st.session_state.mask_prompt = ""
    st.session_state.prompt_text = "Framed photographs over a desk, with a stool under the desk"
def on_2_click():
    st.session_state.masking_mode = 0
    st.session_state.painting_mode = 1
    st.session_state.mask_prompt = ""
    st.session_state.prompt_text = "A cozy living room"
def on_3_click():
    st.session_state.masking_mode = 1
    st.session_state.painting_mode = 0
    st.session_state.mask_prompt = "painting"
    st.session_state.prompt_text = "embedded fireplace"
def on_4_click():
    st.session_state.masking_mode = 1
    st.session_state.painting_mode = 1
    st.session_state.mask_prompt = "painting"
    st.session_state.prompt_text = "living room"

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
            st.write("Mask Mode:",":blue[Image]")
            st.write("Mask Prompt:",":blue[None]")
            st.write("Painting Mode:", ":blue[INPAINTING]")
            st.write("Prompt:",":blue[Framed photographs over a desk, with a stool under the desk]")
            st.button("Load Prompt 1", on_click=on_1_click)    
        with tab2:
            st.write("Mask Mode:",":blue[Image]")
            st.write("Mask Prompt:",":blue[None]")
            st.write("Painting Mode:", ":blue[OUTPAINTING]")
            st.write("Prompt:",":blue[A cozy living room]")
            st.button("Load Prompt 2", on_click=on_2_click)   
        with tab3:
            st.write("Mask Mode:",":blue[Prompt]")
            st.write("Mask Prompt:",":blue[painting]")
            st.write("Painting Mode:", ":blue[INPAINTING]")
            st.write("Prompt:",":blue[embedded fireplace]")
            st.button("Load Prompt 3", on_click=on_3_click)   
        with tab4:
            st.write("Mask Mode:",":blue[Prompt]")
            st.write("Mask Prompt:",":blue[painting]")
            st.write("Painting Mode:", ":blue[OUTPAINTING]")
            st.write("Prompt:",":blue[living room]")
            st.button("Load Prompt 4", on_click=on_4_click)   
            
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
                st.image("images/mask1.png")
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
                        mask_bytes = glib.get_bytes_from_file("images/mask1.png")
                    
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
