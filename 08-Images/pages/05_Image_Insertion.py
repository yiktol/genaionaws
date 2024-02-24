import streamlit as st
import image.image_insertion_lib as glib
from helpers import set_page_config

set_page_config()

st.title("Image Insertion")

col1, col2, col3 = st.columns(3)



placement_options_dict = { #Configure mask areas for image insertion
    "Wall behind desk": (3, 3, 506, 137), #x, y, width, height
    "On top of desk": (78, 60, 359, 115),
    "Beneath desk": (108, 237, 295, 239),
    "Custom": (0, 0, 200, 100), 
}

placement_options = list(placement_options_dict)

if "area" not in st.session_state:
    st.session_state.area = 2
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

prompt_options = [{"pa": "Wall behind desk",
    "prompt": "A painting of a Spanish galleon in rough seas"},
                  {
    "pa": "Wall behind desk",
    "prompt": "Framed family photographs"},
                  {
    "pa": "On top of desk",
    "prompt": "writing instruments"},
                  {
    "pa": "On top of desk",
    "prompt": "A hutch" },
    {
    "pa": "On top of desk",
    "prompt": "A pile of papers"          
    },
    {
    "pa": "Beneath desk",
    "prompt": "a stool"          
    },
    {
    "pa": "Beneath desk",
    "prompt": "A sleeping cat"          
    }
]

def update_options(**kwargs):
    st.session_state.area = placement_options.index(kwargs['area'])
    st.session_state.prompt = kwargs['p']

with col1:
    st.subheader("Initial image")
    
    uploaded_file = st.file_uploader("Select an image (must be 512x512)", type=['png', 'jpg'])
    
    if uploaded_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image("images/desk.jpg")

    container2 = st.container(border=True)    
    with container2:
        st.subheader('Prompt Examples:')
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["P1", "P2", "P3", "P4", "P5", "P6", "P7"])
        with tab1:
            st.write("Placement Area:", prompt_options[0]["pa"])
            st.write("Prompt:", prompt_options[0]["prompt"])
            st.button("Load Prompt", key=0, on_click=update_options, kwargs=dict(area=prompt_options[0]["pa"],p=prompt_options[0]["prompt"]) )     
        with tab2:
            st.write("Placement Area:", prompt_options[1]["pa"])
            st.write("Prompt:", prompt_options[1]["prompt"])
            st.button("Load Prompt", key=1, on_click=update_options, kwargs=dict(area=prompt_options[1]["pa"],p=prompt_options[1]["prompt"]) )    
        with tab3:
            st.write("Placement Area:", prompt_options[2]["pa"])
            st.write("Prompt:", prompt_options[2]["prompt"])
            st.button("Load Prompt", key=2, on_click=update_options, kwargs=dict(area=prompt_options[2]["pa"],p=prompt_options[2]["prompt"]) )   
        with tab4:
            st.write("Placement Area:", prompt_options[3]["pa"])
            st.write("Prompt:", prompt_options[3]["prompt"])
            st.button("Load Prompt", key=3, on_click=update_options, kwargs=dict(area=prompt_options[3]["pa"],p=prompt_options[3]["prompt"]) )  
        with tab5:
            st.write("Placement Area:", prompt_options[4]["pa"])
            st.write("Prompt:", prompt_options[4]["prompt"])
            st.button("Load Prompt", key=4, on_click=update_options, kwargs=dict(area=prompt_options[4]["pa"],p=prompt_options[4]["prompt"]) )    
        with tab6:
            st.write("Placement Area:", prompt_options[5]["pa"])
            st.write("Prompt:", prompt_options[5]["prompt"])
            st.button("Load Prompt", key=5, on_click=update_options, kwargs=dict(area=prompt_options[5]["pa"],p=prompt_options[5]["prompt"]) )    
        with tab7:
            st.write("Placement Area:", prompt_options[6]["pa"])
            st.write("Prompt:", prompt_options[6]["prompt"])
            st.button("Load Prompt", key=6, on_click=update_options, kwargs=dict(area=prompt_options[6]["pa"],p=prompt_options[6]["prompt"]) )    




with col2:
    st.subheader("Insertion parameters")
    
    with st.form("form1"):
        placement_area = st.radio("Placement area:", ['Wall behind desk', 'On top of desk', 'Beneath desk', 'Custom'], index = st.session_state.area)
        with st.expander("Custom:", expanded=False):
            
            mask_dimensions = placement_options_dict[placement_area]
        
            mask_x = st.number_input("Mask x position", value=mask_dimensions[0])
            mask_y = st.number_input("Mask y position", value=mask_dimensions[1])
            mask_width = st.number_input("Mask width", value=mask_dimensions[2])
            mask_height = st.number_input("Mask height", value=mask_dimensions[3])
        
        prompt_text = st.text_area("Object to add:", height=100, key="prompt", help="The prompt text")
        
        generate_button = st.form_submit_button("Generate", type="primary")
    

with col3:
    st.subheader("Result")

    if generate_button:
        with st.spinner("Drawing..."):
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
            else:
                image_bytes = None
            
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                image_bytes=image_bytes, 
                insertion_position=(mask_x, mask_y),
                insertion_dimensions=(mask_width, mask_height),
            )
        
        st.image(generated_image)
