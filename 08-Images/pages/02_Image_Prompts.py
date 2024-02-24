import streamlit as st
import image.image_prompts_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Prompts")

col1, col2 = st.columns(2)

if "element" not in st.session_state:
    st.session_state.element = "Subject"
if "prompt" not in st.session_state:
    st.session_state.prompt = "Doctor"
if "negative_prompt" not in st.session_state:
    st.session_state.negative_prompt = ""

options = [{"element":"Subject","prompt": "Doctor", "negative_prompt": ""},
           {"element":"Medium","prompt": 'Painting of a doctor', "negative_prompt": ""},
           {"element":"Style","prompt": 'Painting of a doctor, Impressionist style', "negative_prompt": ""},
           {"element":"Shot type/angle","prompt": 'Painting of a doctor, Impressionist style, low-angle shot', "negative_prompt": ""},
           {"element":"Light","prompt": "Painting of a doctor, Impressionist style, low-angle shot, dim lighting", "negative_prompt": ""},
           {"element":"Color scheme","prompt": "Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme", "negative_prompt": ""},
           {"element":"Negative prompt","prompt": "Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme", "negative_prompt": "Stethoscope"},
           {"element":"Text","prompt": "Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme, sign reading \"The Doctor is in\"", "negative_prompt": "Stethoscope"},
]

def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.negative_prompt  = options[item_num]["negative_prompt"]

def load_options(item_num):
    st.write("Element added:", options[item_num]["element"])
    st.write("Prompt:", options[item_num]["prompt"])
    st.write("Negative Prompt:", options[item_num]["negative_prompt"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(item_num,))
    
with col1:
    st.subheader("Image parameters")
    
    with st.form("form1"):
        prompt_text = st.text_area("What you want to see in the image:", key="prompt",height=100, help="The prompt text")
        negative_prompt = st.text_input("What shoud not be in the image:",key="negative_prompt", help="The negative prompt")
        generate_button = st.form_submit_button("Generate", type="primary")

    container2 = st.container(border=True)    
    with container2:
        st.subheader('Prompt Examples:')
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6", "Prompt7", "Prompt8"])
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
        with tab6:
            load_options(item_num=5)
        with tab7:
            load_options(item_num=6)
        with tab8:
            load_options(item_num=7)

with col2:
    st.subheader("Result")

    if generate_button:
        with st.spinner("Drawing..."):
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                negative_prompt=negative_prompt,
            )
        
        st.image(generated_image)
