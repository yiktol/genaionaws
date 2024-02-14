import streamlit as st
import json
import boto3
import math
import pandas as pd
from helpers import get_embedding, calculate_dot_product, bedrock_runtime_client, set_page_config

set_page_config()

bedrock = bedrock_runtime_client()

st.title("Dot Product")

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("dot-product.svg")
    st.latex(r'a.b = |a|  x  |b|  x  \cos \theta') 

st.write("""Dot product similarity is another commonly used method to measure similarity. \
In linear algebra, the dot product between two vectors measures the extent to which two vectors align in the same direction. \
If the dot product of two vectors is 0, the two vectors are orthogonal (perpendicular), which is sort of an intermediate similarity.""")

with st.form("myform"):
    prompt = st.text_area(":orange[Enter your prompt here:]", height = 50, value="Hello"),
    text1=st.text_input('Text1',value="Hi"),
    text2=st.text_input('Text2',value="Good Day"),
    text3=st.text_input('Text3',value="How are you"),
    text4=st.text_input('Text4',value="What is general relativity"),
    text5=st.text_input('Text5',value="She sells sea shells on the sea shore"),
    submit = st.form_submit_button("Calculate Dot Product",type="primary")

txt_array=[]
distance_array=[]

#print(text1)
if prompt and submit:
    prompt_embedding = get_embedding(bedrock, prompt[0])
    
    texts = [text1, text2, text3, text4, text5]
    for text in texts:
        embedding = get_embedding(bedrock, text[0])
        distance = calculate_dot_product(prompt_embedding, embedding)
        txt_array.append(text[0])
        distance_array.append(distance)
        #print(distance)
        
        
    df = pd.DataFrame({'Text':txt_array, 'Dot Product':distance_array})
    st.table(df)