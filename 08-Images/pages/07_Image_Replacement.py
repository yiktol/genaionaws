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

options = [{"image": "images/example3.jpg","mask_prompt": "Pink curtains","prompt_text": "Green curtains"},
           {"image": "images/example3.jpg","mask_prompt": "Table","prompt_text": ""},
           {"image": "images/z1034.jpg","mask_prompt": "Toy house","prompt_text": "Log cabin"},
           {"image": "images/desk1.jpg","mask_prompt": "Stool","prompt_text": ""}
    ]


def update_options(item_num):
    st.session_state.image = options[item_num]["image"]
    st.session_state.mask_prompt = options[item_num]["mask_prompt"]
    st.session_state.prompt_text = options[item_num]["prompt_text"]

def load_options(item_num):
    st.write("Image:",options[item_num]["image"])
    st.write("Mask Prompt:",options[item_num]["mask_prompt"])
    st.write("Prompt:",options[item_num]["prompt_text"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(item_num,))
    
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
            load_options(item_num=0)       
        with tab2:
            load_options(item_num=1)            
        with tab3:
            load_options(item_num=2)                      
        with tab4:
            load_options(item_num=3)                      
           
            
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
