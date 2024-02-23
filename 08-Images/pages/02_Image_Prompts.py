import streamlit as st
import image.image_prompts_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Prompts")

col1, col2 = st.columns(2)


with col1:
    st.subheader("Image parameters")
    
    with st.form("form1"):
        prompt_text = st.text_area("What you want to see in the image:", height=100, help="The prompt text")
        negative_prompt = st.text_input("What shoud not be in the image:", help="The negative prompt")
        generate_button = st.form_submit_button("Generate", type="primary")

with col2:
    st.subheader("Result")

    if generate_button:
        with st.spinner("Drawing..."):
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                negative_prompt=negative_prompt,
            )
        
        st.image(generated_image)
