import streamlit as st
import image.image_replacement_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Replacement")

col1, col2, col3 = st.columns(3)

if "image" not in st.session_state:
    st.session_state.image = "images/example3.jpg"
if "mask_prompt" not in st.session_state:
    st.session_state.mask_prompt = "Pink curtains"
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = "Green curtains"


def on_1_click():
    st.session_state.image = "images/example3.jpg"
    st.session_state.mask_prompt = "Pink curtains"
    st.session_state.prompt_text = "Green curtains"
def on_2_click():
    st.session_state.image = "images/example3.jpg"
    st.session_state.mask_prompt = "Table"
    st.session_state.prompt_text = ""
def on_3_click():
    st.session_state.image = "images/z1034.jpg"
    st.session_state.mask_prompt = "Toy house"
    st.session_state.prompt_text = "Log cabin"
def on_4_click():
    st.session_state.image = "images/desk1.jpg"
    st.session_state.mask_prompt = "Stool"
    st.session_state.prompt_text = ""
    
with col1:
    st.subheader("Image parameters")
    
    uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])
    
    if uploaded_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image(st.session_state.image)
  
    container2 = st.container(border=True)    
    with container2:
        st.subheader('Prompt Examples:')
        tab1, tab2, tab3, tab4 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4"])
        with tab1:
            st.write("Image:",":blue[images/example3.jpg]")
            st.write("Mask Prompt:",":blue[Pink curtains]")
            st.write("Prompt:",":blue[Green curtains]")
            st.button("Load Prompt 1", on_click=on_1_click)          
        with tab2:
            st.write("Image:",":blue[images/example3.jpg]")
            st.write("Mask Prompt:",":blue[Table]")
            st.write("Prompt:",":blue[None]")
            st.button("Load Prompt 2", on_click=on_2_click)        
        with tab3:
            st.write("Image:",":blue[images/z1034.jpg]")
            st.write("Mask Prompt:",":blue[Toy house]")
            st.write("Prompt:",":blue[Log cabin]")
            st.button("Load Prompt 3", on_click=on_3_click)                 
        with tab4:
            st.write("Image:",":blue[images/desk1.jpg]")
            st.write("Mask Prompt:",":blue[Stool]")
            st.write("Prompt:",":blue[None]")
            st.button("Load Prompt 4", on_click=on_4_click)                 
           
            
with col2:
    st.subheader("Image Prompts:")
    with st.form("form1"):
        mask_prompt = st.text_input("Object to remove/replace", key="mask_prompt", help="The mask text")
        
        prompt_text = st.text_area("Object to add (leave blank to remove)", key="prompt_text", height=100, help="The prompt text")
        
        generate_button = st.form_submit_button("Generate", type="primary")

with col3:
    st.subheader("Result")

    if generate_button:
        with st.spinner("Drawing..."):
            
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
            else:
                image_bytes = glib.get_bytes_from_file(st.session_state.image)
            
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                image_bytes=image_bytes, 
                mask_prompt=mask_prompt,
            )
        
        st.image(generated_image)
