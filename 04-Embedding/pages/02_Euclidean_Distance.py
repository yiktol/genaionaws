import streamlit as st
import json
import boto3
import math
import pandas as pd
from helpers import get_embedding, calculate_distance, bedrock_runtime_client, set_page_config

set_page_config()

bedrock = bedrock_runtime_client()

st.title("Euclidean Distance")

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("dist-2-points-3d.svg")
    st.latex(r'd(p,q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}')

st.write("""When you use vectors to represent text, you can calculate the Euclidean distance between two pieces of text. \
You might wonder why people care about “the distance between two pieces of text”. What does it really mean?\n
The Euclidean distance between two pieces of text indicates how far away they are from each other. \
In human readable terms, it means how similar they are in natural language. \
If the distance is very small, the two pieces of text convey a similar message. \
If the distance is very big, the two pieces of text convey a different message.""")

with st.form("myform"):
    prompt = st.text_area(":orange[Enter your prompt here:]", height = 50, value="Hello"),
    text1=st.text_input('Text1',value="Hi"),
    text2=st.text_input('Text2',value="Good Day"),
    text3=st.text_input('Text3',value="How are you"),
    text4=st.text_input('Text4',value="What is general relativity"),
    text5=st.text_input('Text5',value="She sells sea shells on the sea shore"),
    submit = st.form_submit_button("Check Distance",type="primary")

txt_array=[]
distance_array=[]

#print(text1)
if prompt and submit:
    prompt_embedding = get_embedding(bedrock, prompt[0])
    
    texts = [text1, text2, text3, text4, text5]
    for text in texts:
        embedding = get_embedding(bedrock, text[0])
        distance = calculate_distance(prompt_embedding, embedding)
        txt_array.append(text[0])
        distance_array.append(distance)
        #print(distance)
        
        
    df = pd.DataFrame({'Text':txt_array, 'Distance':distance_array})
    st.table(df)