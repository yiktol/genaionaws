import streamlit as st #all streamlit commands will be available through the "st" alias
import image.image_lib as glib #reference to local lib script
from helpers import set_page_config

set_page_config() #set the page width wider to accommodate columns

st.title("Image Generation") #page title

col1, col2 = st.columns(2) #create 2 columns

prompt_text = "rocket ship launching from forest with flower garden under a blue sky, masterful, ghibli" #initialize prompt text

with col1: #everything in this with block will be placed in column 1
    st.subheader("Prompt:") #subhead for this column
    
    with st.form("form1"):
        prompt_text = st.text_area("Prompt", prompt_text, height=200, label_visibility="collapsed") #display a multiline text box with no label
        process_button = st.form_submit_button("Run", type="primary") #display a primary button


with col2: #everything in this with block will be placed in column 2
    st.subheader("Result:") #subhead for this column
    
    if process_button: #code in this if block will be run when the button is clicked
        with st.spinner("Drawing..."): #show a spinner while the code in this with block runs
            generated_image = glib.get_image_response(prompt_content=prompt_text) #call the model through the supporting library
        
        st.image(generated_image) #display the generated image


