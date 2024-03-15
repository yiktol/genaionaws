import boto3
import time
import streamlit as st
import numpy as np
import pandas as pd
import utils.helpers as helpers
import utils.titan_multimodal as titan_multimodal
from PIL import Image
from io import BytesIO


helpers.set_page_config()
# helpers.reset_session()
#Create the connection to Bedrock
bedrock_runtime = helpers.bedrock_runtime_client()


if "multimodal_embeddings" not in st.session_state:
    st.session_state.multimodal_embeddings = []
    
if "distance" not in st.session_state:
    st.session_state.distance = np.array([])
    
if "is_multimodal_embeddings" not in st.session_state:
    st.session_state.is_multimodal_embeddings = False

text, code = st.columns([0.6, 0.4])

dataset = helpers.load_jsonl('data/metadata.jsonl')
sttable = {}

products = []
for i in range(len(dataset)):
    item = dataset[i]['file_name'].split('/')[-1].split('.')[0].lower()
    item = item.replace('_', ' ')
    item = item[0].upper() + item[1:]
    products.append(item)

image_temp = st.empty()

def generate_embeddings():
    with st.spinner("Generating Embeddings for all products..."):
        multimodal_embeddings = []
        for i in range(len(dataset)):
            embedding = titan_multimodal.embedding(image_path=dataset[i]['file_name'], dimension=1024)["embedding"]
            multimodal_embeddings.append(embedding)
            # print(f"generated embedding for {dataset[i]['file_name']}") 
            # image_temp = dataset[i]['file_name']
        st.session_state.multimodal_embeddings = multimodal_embeddings
        st.session_state.is_multimodal_embeddings =  True
    st.success("Embeddings Generated!, you may now search.")
    time.sleep(5)
    st.rerun()
    
def display_embeddings_by_index(idx):
    # st.write(f"Embeddings for product: {products[idx]}")
    temp_embedding = st.session_state.multimodal_embeddings[idx]
    df = display_embeddings(temp_embedding)
    return df
    
def display_embeddings(embedding):
    numbers = (np.array(embedding).reshape(32,32))
    df = pd.DataFrame(numbers, columns=("col %d" % i for i in range(32)))
    return df


with text:
    st.header("Multimodal Embedding and Searching")
    st.write("""Amazon Titan Multimodal Embedding Models can be used for enterprise tasks such as image search and similarity based recommendation, and has built-in mitigation that helps reduce bias in searching results. \
There are multiple embedding dimension sizes for best latency/accuracy tradeoffs for different needs, and all can be customized with a simple API to adapt to your own data while persists data security and privacy.""")
    
    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area(":orange[Search for a product:]", value="suede sneaker")
        k = st.number_input("Number of results", value=1, min_value=1, max_value=10)
        submit = st.form_submit_button("Search",type="primary")   

        
    if submit:
        if st.session_state.is_multimodal_embeddings == False:
            st.warning("No Embeddings available, please Generate Embeddings.")
        else:
            text_embedding = titan_multimodal.embedding(description=prompt_data)["embedding"]
            with st.expander("See Text Embedding"):
                st.dataframe(display_embeddings(text_embedding),use_container_width=True,height=500)
            st.write("Result")
            idx_returned, distance = titan_multimodal.multimodal_search(
                description=prompt_data,
                multimodal_embeddings=st.session_state.multimodal_embeddings,
                top_k=k)
            for idx in idx_returned[:]:
                st.image(Image.open(f"{dataset[idx]['file_name']}"))
            
            sttable = {"Product Name":"Distance"}
            distance_list = np.array(distance).tolist()
            for product in products:
                sttable[product] = distance_list[0][products.index(product)] 
            

    
with code:
       
    with st.container(border=True):       
        col1, col2 = code.columns(2)
        # st.subheader("Products")
        option = st.selectbox("Select a Product:",products)
        idx = products.index(option)
        st.image(dataset[idx]['file_name'], caption=dataset[idx]['description'])
        
        generate = col1.button("Generate Embeddings")
        clear_embeddings = col2.button("Clear Embeddings", on_click=helpers.reset_session)
    
        if st.session_state.is_multimodal_embeddings == True:
            with st.expander("See Image Embedding"):
                st.dataframe(display_embeddings_by_index(idx),use_container_width=True,height=500)
        
        
    if generate:
        if st.session_state.multimodal_embeddings and st.session_state.is_multimodal_embeddings:
            st.warning("Embeddings already generated, you can search a product.")
        else:
            generate_embeddings()
    
    if sttable:
        code.dataframe(sttable, use_container_width=True, height=500) 