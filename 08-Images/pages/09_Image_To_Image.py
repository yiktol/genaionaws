import streamlit as st
import image.image_to_image_lib as glib

#

st.set_page_config(layout="wide", page_title="Image to Image")

st.title("Image to Image")

col1, col2 = st.columns(2)

#

with col1:
    st.subheader("Input image")
    
    # uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])
    
    # if uploaded_file:
    #     uploaded_image_preview = glib.get_resized_image_io(uploaded_file.getvalue())
    #     st.image(uploaded_image_preview)
    # else:
    #     with open('samples/flower2.jpg', 'rb') as f:
    #         file_bytes = f.read()
    #     st.image('samples/flower2.jpg')    
    
    prompt_text = st.text_area("Prompt text", 
                               "rocket ship launching from forest with flower garden under a blue sky, masterful, ghibli",
                               height=200)
    
    process_button = st.button("Generate", type="primary")
    
    st.subheader("Result")

    if process_button:
        with st.spinner("Drawing..."):
            generated_image = glib.get_image_response(prompt_content=prompt_text)
            # else:
            #     generated_image = glib.get_altered_image_from_model(prompt_content=prompt_text, image_bytes=file_bytes)  
        
        st.image(generated_image)
    else:
        st.image('generated_image.png')

with col2:
    st.subheader("New image")
    prompt_text2 = st.text_area("Modify Image Prompt", 
                            "crayon drawing of rocket ship launching from forest",
                            height=200)
    
    process_button2 = st.button("Modify", type="primary")
    st.subheader("Result")

    if process_button2:
        with st.spinner("Drawing..."):
            with open('generated_image.png', 'rb') as f:
                init_image = f.read()
            generated_image2 = glib.get_altered_image_from_model(prompt_content=prompt_text2, image_bytes=init_image)
            # else:
            #     generated_image = glib.get_altered_image_from_model(prompt_content=prompt_text, image_bytes=file_bytes)  
        
        st.image(generated_image2)
