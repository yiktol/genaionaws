import streamlit as st #all streamlit commands will be available through the "st" alias
import image.image_search_lib as glib #reference to local lib script
from io import BytesIO
import PIL.Image as Image
from helpers import set_page_config

set_page_config()
st.title("Image Search") #page title


if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
    with st.spinner("Indexing images..."): #show a spinner while the code in this with block runs
        st.session_state.vector_index = glib.get_index() #retrieve the index through the supporting library and store in the app's session cache


search_images_tab, find_similar_images_tab = st.tabs(["Image search", "Find similar images"])

if 'query' not in st.session_state:
    st.session_state.query = ''

def on_house_click():
    st.session_state.query = 'house'
def on_woman_click():
    st.session_state.query = 'woman'
def on_toys_click():
    st.session_state.query = 'toys'
def on_animal_click():
    st.session_state.query = 'animals'
    
ct = st.container(border=False)

with search_images_tab:

    search_col_1, search_col_2 = st.columns(2)
    
    col1, col2, col3, col4 = search_col_1.columns(4)
    
    with search_col_1:
        with col1:
            st.write(":orange[Load Search:]")
        with col2:
            st.button('house', on_click=on_house_click)
            st.button('animals', on_click=on_animal_click)
        with col3:
            st.button('woman', on_click=on_woman_click)
        with col4:
            st.button('toys', on_click=on_toys_click)
        with st.form("search_form"): #create a form with a unique name (search_form)
            input_text = st.text_input("Search for:", key="query") #display a multiline text box with no label
            search_button = st.form_submit_button("Search", type="primary") #display a primary button
        
    with search_col_2:
        if search_button: #code in this if block will be run when the button is clicked
            st.subheader("Results")
            with st.spinner("Searching..."): #show a spinner while the code in this with block runs
                response_content = glib.get_similarity_search_results(index=st.session_state.vector_index, search_term=input_text)
                
                for res in response_content:
                    st.image(res, width=250)


with find_similar_images_tab:
    
    find_col_1, find_col_2 = st.columns(2)

    with find_col_1:
    
        uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'])
        
        if uploaded_file:
            uploaded_image_preview = uploaded_file.getvalue()
            st.image(uploaded_image_preview)
        else:
            with open('samples/house.jpg', 'rb') as f:
                file_bytes = f.read()
            st.image('samples/house.jpg')    
    
        find_button = st.button("Find", type="primary") #display a primary button
        


    with find_col_2:
        if find_button: #code in this if block will be run when the button is clicked
            st.subheader("Results")
            with st.spinner("Finding..."): #show a spinner while the code in this with block runs
                if uploaded_file:
                    response_content = glib.get_similarity_search_results(index=st.session_state.vector_index, search_image=uploaded_file.getvalue())
                if not uploaded_file:
                    response_content = glib.get_similarity_search_results(index=st.session_state.vector_index, search_image=file_bytes)
                #st.write(response_content) #using table so text will wrap
                
                for res in response_content:
                    st.image(res, width=250)
    
            
        