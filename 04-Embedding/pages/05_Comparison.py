import streamlit as st
import json
import boto3
import math
import pandas as pd
from helpers import get_embedding, calculate_cosine_similarity,calculate_distance,calculate_dot_product, bedrock_runtime_client, set_page_config

set_page_config()

bedrock = bedrock_runtime_client()

st.header("Similarity Metrics")

left_co, cent_co,last_co = st.columns([0.3,0.3,0.4])
with left_co:
    st.write('Cosine Similarity ')
    st.latex(r'\cos \theta = \frac{a.b}  {|a| x |b|}')
    st.image("dot-product-a-cos.svg")
with cent_co:
    st.write('Dot Product')
    st.latex(r'a.b = |a|  x  |b|  x  \cos \theta') 
    st.image("dot-product.svg")  
with last_co:
    st.write('Euclidean Distance')
    st.latex(r'd(p,q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}')
    st.image("dist-2-points-a.svg")
    

st.write("""Similarity metrics are a vital tool in many data analysis and machine learning tasks,\
allowing us to compare and evaluate the similarity between different pieces of data. \
Many different metrics are available, each with pros and cons and suitable for different data types and tasks.
""")

with st.form("myform"):
    prompt = st.text_area(":orange[Enter your prompt here:]", height = 50, value="Hello"),
    text1=st.text_input('Text1',value="Hi"),
    text2=st.text_input('Text2',value="Good Day"),
    text3=st.text_input('Text3',value="How are you"),
    text4=st.text_input('Text4',value="What is general relativity"),
    text5=st.text_input('Text5',value="She sells sea shells on the sea shore"),
    submit = st.form_submit_button("Compare",type="primary")

txt_array=[]
distance_array=[]
dot_product_array=[]
similarity_array=[]

#print(text1)
if prompt and submit:
    prompt_embedding = get_embedding(bedrock, prompt[0])
    
    texts = [text1, text2, text3, text4, text5]
    for text in texts:
        embedding = get_embedding(bedrock, text[0])
        similarity = calculate_cosine_similarity(prompt_embedding, embedding)
        distance = calculate_distance(prompt_embedding, embedding)
        dot_product = calculate_dot_product(prompt_embedding, embedding)
        txt_array.append(text[0])
        similarity_array.append(similarity)
        distance_array.append(distance)
        dot_product_array.append(dot_product)
        #print(distance)
        
        
    df = pd.DataFrame({'Text':txt_array, 'Cosine Similarity':similarity_array, 'Dot Product':dot_product_array, 'Euclidean Distance':distance_array})
    st.table(df)