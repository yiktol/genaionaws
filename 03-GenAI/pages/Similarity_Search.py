import streamlit as st #all streamlit commands will be available through the "st" alias
import search_lib as glib #reference to local lib script
from helpers import set_page_config

set_page_config()

st.title("Embeddings Search") #page title
st.write("Facebook AI Similarity Search (Faiss), a library that allows us to quickly search for multimedia documents that are similar to each other — a challenge where traditional query search engines fall short. ")

if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
    with st.spinner("Indexing document..."): #show a spinner while the code in this with block runs
        st.session_state.vector_index = glib.get_index() #retrieve the index through the supporting library and store in the app's session cache


input_text = st.text_input("Ask a question about Amazon Bedrock:") #display a multiline text box with no label
go_button = st.button("Go", type="primary") #display a primary button


if go_button: #code in this if block will be run when the button is clicked
    
    with st.spinner("Working..."): #show a spinner while the code in this with block runs
        response_content = glib.get_similarity_search_results(index=st.session_state.vector_index, question=input_text)
        
        st.table(response_content) #using table so text will wrap