import streamlit as st
import pandas as pd
import utils.helpers as helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()

text, code = st.columns([0.6, 0.4])

with text:
    st.header("Dot Product")
    col1, col2, col3 = text.columns([0.3, 0.4,0.3])
    col2.image("dot-product.svg")
    st.latex(r'a.b = |a|  x  |b|  x  \cos \theta') 

    st.write("""Dot product similarity is another commonly used method to measure similarity. \
    In linear algebra, the dot product between two vectors measures the extent to which two vectors align in the same direction. \
    If the dot product of two vectors is 0, the two vectors are orthogonal (perpendicular), which is sort of an intermediate similarity.""")

    with st.form("myform"):
        prompt = st.text_area(":orange[Enter your prompt here:]", height = 50, value="Hello"),
        text1=st.text_area('Text1',value="Hi"),
        text2=st.text_area('Text2',value="Good Day"),
        text3=st.text_area('Text3',value="How are you"),
        text4=st.text_area('Text4',value="What is general relativity"),
        text5=st.text_area('Text5',value="She sells sea shells on the sea shore"),
        submit = st.form_submit_button("Calculate Dot Product",type="primary")

    txt_array=[]
    distance_array=[]
    prompt_display = []
    #print(text1)
    if prompt and submit:
        with st.spinner("Calculating Dot Product..."):
            prompt_embedding = helpers.get_embedding(bedrock, prompt[0])
            
            texts = [text1, text2, text3, text4, text5]
            for text in texts:
                embedding = helpers.get_embedding(bedrock, text[0])
                distance = helpers.calculate_dot_product(prompt_embedding, embedding)
                txt_array.append(text[0])
                distance_array.append(distance)
                prompt_display.append(prompt[0])
                
                
            df = pd.DataFrame({'Prompt':prompt_display,'Text':txt_array, 'Dot Product':distance_array})
            st.subheader("Dot Product between :orange[Prompt] and :orange[Text]")
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

def calculate_dot_product_similarity(v1, v2):
    similarity = dot(v1, v2)
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
    distance = calculate_dot_product_similarity(hello, embedding)
    print(distance)
    """
    st.subheader("Code")
    st.code(code_data, language="python")

