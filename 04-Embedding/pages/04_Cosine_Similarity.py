import streamlit as st
import json
import boto3
import math
import pandas as pd
from helpers import get_embedding, calculate_cosine_similarity, bedrock_runtime_client, set_page_config

set_page_config()

bedrock = bedrock_runtime_client()

text, code = st.columns(2)

with text:
    st.title("Cosine Similarity")
    st.image("dot-product-a-cos.svg")
    st.latex(r'\cos \theta = \frac{a.b}  {|a| x |b|}') 

    st.write("""Cosine Similarity is another commonly used method to measure similarity.\
    In linear algebra, cosine similarity is the cosine of the angle between two vectors.\
    That is, it is the dot product of the vectors divided by the product of their lengths. \
    Similar to Euclidean distance, cosine similarity measures how similar two pieces of text are likely to be in terms of their subject matter,\
    which is independent of the length of the texts. When cosine similarity equals to 1,\
    it means two pieces of text have identical meanings in natural language.
    """)

    with st.form("myform"):
        prompt = st.text_area(":orange[Enter your prompt here:]", height = 50, value="Hello"),
        text1=st.text_input('Text1',value="Hi"),
        text2=st.text_input('Text2',value="Good Day"),
        text3=st.text_input('Text3',value="How are you"),
        text4=st.text_input('Text4',value="What is general relativity"),
        text5=st.text_input('Text5',value="She sells sea shells on the sea shore"),
        submit = st.form_submit_button("Check Similarity",type="primary")

    txt_array=[]
    distance_array=[]

    #print(text1)
    if prompt and submit:
        prompt_embedding = get_embedding(bedrock, prompt[0])
        
        texts = [text1, text2, text3, text4, text5]
        for text in texts:
            embedding = get_embedding(bedrock, text[0])
            distance = calculate_cosine_similarity(prompt_embedding, embedding)
            txt_array.append(text[0])
            distance_array.append(distance)
            #print(distance)
            
            
        df = pd.DataFrame({'Text':txt_array, 'Similarity':distance_array})
        st.table(df)

with code:
    
    code_data=f"""import json
import boto3
from numpy import dot
from numpy.linalg import norm

def get_embedding(bedrock, text):
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    input = {{
            'inputText': text
         }}

    response = bedrock.invoke_model(
        body=json.dumps(input), 
        modelId=modelId, 
        accept=accept,
        contentType=contentType
        )
        
    response_body = json.loads(response.get('body').read())
    embedding = response_body['embedding']
    return embedding

def calculate_cosine_similarity(v1, v2):
    similarity = dot(v1, v2)/(norm(v1)*norm(v2))
    return similarity

# main function
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )

hello = get_embedding(bedrock, \"{prompt[0]}\")
texts = [
    \"{text1[0]}\",
    \"{text2[0]}\",
    \"{text3[0]}\",
    \"{text4[0]}\",
    \"{text5[0]}\"
    ]
    
for text in texts:
    embedding = get_embedding(bedrock, text)
    distance = calculate_cosine_similarity(hello, embedding)
    print(distance)
    """
    st.code(code_data, language="python")