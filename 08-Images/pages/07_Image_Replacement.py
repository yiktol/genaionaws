import streamlit as st
import image.image_replacement_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Replacement")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Image parameters")
    
    uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])
    
    if uploaded_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image("images/example3.jpg")
    
    
with col2:
    with st.form("form1"):
        mask_prompt = st.text_input("Object to remove/replace", value="Pink curtains", help="The mask text")
        
        prompt_text = st.text_area("Object to add (leave blank to remove)", value="Green curtains", height=100, help="The prompt text")
        
        generate_button = st.form_submit_button("Generate", type="primary")
    
    
with col3:
    st.subheader("Result")

    if generate_button:
        with st.spinner("Drawing..."):
            
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
            else:
                image_bytes = glib.get_bytes_from_file("images/example3.jpg")
            
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                image_bytes=image_bytes, 
                mask_prompt=mask_prompt,
            )
        
        st.image(generated_image)
