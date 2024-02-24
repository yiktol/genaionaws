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

options = [{"pa": "Wall behind desk","prompt": "A painting of a Spanish galleon in rough seas"},
                  {"pa": "Wall behind desk","prompt": "Framed family photographs"},
                  {"pa": "On top of desk","prompt": "writing instruments"},
                  {"pa": "On top of desk","prompt": "A hutch" },
                  {"pa": "On top of desk","prompt": "A pile of papers"},
                  {"pa": "Beneath desk","prompt": "a stool"},
                  {"pa": "Beneath desk","prompt": "A sleeping cat"}
                ]

def update_options(**kwargs):
    st.session_state.area = placement_options.index(kwargs['area'])
    st.session_state.prompt = kwargs['p']

def load_options(item_num):
    st.write("Placement Area:", options[item_num]["pa"])
    st.write("Prompt:", options[item_num]["prompt"])
    st.button("Load Prompt", key=item_num, on_click=update_options, kwargs=dict(area=options[item_num]["pa"],p=options[item_num]["prompt"]) ) 

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
