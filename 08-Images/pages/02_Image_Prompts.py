import streamlit as st
import image.image_prompts_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Prompts")

col1, col2 = st.columns(2)

def on_1_click():
    st.session_state.prompt = "Doctor"
    st.session_state.negative_prompt  = ""
def on_2_click():
    st.session_state.prompt  = 'Painting of a doctor'
    st.session_state.negative_prompt  = ""
def on_3_click():
    st.session_state.prompt  = 'Painting of a doctor, Impressionist style'
    st.session_state.negative_prompt  = ""
def on_4_click():
    st.session_state.prompt  = 'Painting of a doctor, Impressionist style, low-angle shot'
    st.session_state.negative_prompt  = ""
def on_5_click():
    st.session_state.prompt  = "Painting of a doctor, Impressionist style, low-angle shot, dim lighting"
    st.session_state.negative_prompt  = ""
def on_6_click():
    st.session_state.prompt  = "Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme"
    st.session_state.negative_prompt  = ""
def on_7_click():
    st.session_state.prompt  = "Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme"
    st.session_state.negative_prompt  = "Stethoscope"
def on_8_click():
    st.session_state.prompt  = "Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme, sign reading \"The Doctor is in\""   
    st.session_state.negative_prompt  = "Stethoscope" 
    

    
with col1:
    st.subheader("Image parameters")
    
    with st.form("form1"):
        prompt_text = st.text_area("What you want to see in the image:", key="prompt",height=100, help="The prompt text")
        negative_prompt = st.text_input("What shoud not be in the image:",key="negative_prompt", help="The negative prompt")
        generate_button = st.form_submit_button("Generate", type="primary")

    st.subheader('Prompt Examples:')
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6", "Prompt7", "Prompt8"])
    with tab1:
        st.write("Element added:",":orange[Subject]")
        st.write("Prompt:",":blue[Doctor]")
        st.button("Load Prompt 1", on_click=on_1_click)
    with tab2:
        st.write("Element added:",":orange[Medium]")
        st.write("Prompt:",":blue[Painting of a doctor]")
        st.button("Load Prompt 2", on_click=on_2_click)
    with tab3:
        st.write("Element added:",":orange[Style]")
        st.write("Prompt:",":blue[Painting of a doctor, Impressionist style]")
        st.button("Load Prompt 3", on_click=on_3_click)
    with tab4:
        st.write("Element added:",":orange[Shot type/angle]")
        st.write("Prompt:",":blue[Painting of a doctor, Impressionist style, low-angle shot]")
        st.button("Load Prompt 4", on_click=on_4_click)
    with tab5:
        st.write("Element added:",":orange[Light]")
        st.write("Prompt:",":blue[Painting of a doctor, Impressionist style, low-angle shot, dim lighting]")
        st.button("Load Prompt 5", on_click=on_5_click)
    with tab6:
        st.write("Element added:",":orange[Color scheme]")
        st.write("Prompt:",":blue[Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme]")
        st.button("Load Prompt 6", on_click=on_6_click)
    with tab7:
        st.write("Element added:",":orange[Negative prompt]")
        st.write("Prompt:",":blue[Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme]")
        st.write("Negative Prompt:", "Stethoscope")
        st.button("Load Prompt 7", on_click=on_7_click)
    with tab8:
        st.write("Element added:",":orange[Text]")
        st.write("Prompt:",":blue[Painting of a doctor, Impressionist style, low-angle shot, dim lighting, blue and purple color scheme, sign reading \"The Doctor is in\"]")
        st.write("Negative Prompt:", "Stethoscope")
        st.button("Load Prompt 8", on_click=on_8_click)

with col2:
    st.subheader("Result")

    if generate_button:
        with st.spinner("Drawing..."):
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                negative_prompt=negative_prompt,
            )
        
        st.image(generated_image)
