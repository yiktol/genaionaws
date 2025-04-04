import streamlit as st
import pandas as pd
import numpy as np
import utils.helpers as helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()

st.header("Dot Product")
col1, col2, col3 = st.columns([0.2, 0.1,0.7])
col1.image("dot-product.svg")
col2.latex(r'a.b = |a|  x  |b|  x  \cos \theta') 

st.write("""Dot product similarity is another commonly used method to measure similarity. \
In linear algebra, the dot product between two vectors measures the extent to which two vectors align in the same direction. \
If the dot product of two vectors is 0, the two vectors are orthogonal (perpendicular), which is sort of an intermediate similarity.""")

df = None

text, code = st.columns([0.4, 0.6])

with text:

    with st.form("myform"):
        prompt = st.text_area(":orange[Enter your prompt here:]", height = 68, value="Hello"),
        text1=st.text_area('Text1',value="Hi"),
        text2=st.text_area('Text2',value="Good Day"),
        text3=st.text_area('Text3',value="How are you"),
        text4=st.text_area('Text4',value="Good Morning"),
        text5=st.text_area('Text5',value="Goodbye"),
        submit = st.form_submit_button("Calculate Dot Product",type="primary")

    txt_array=[]
    distance_array=[]
    prompt_display = []
    embedding_list =[]
    prompt_embedding = []
    
    if prompt and submit:
        with st.spinner("Calculating Dot Product..."):
            prompt_embedding = helpers.get_embedding(bedrock, prompt[0])
            
            texts = [text1, text2, text3, text4, text5]
            for text in texts:
                embedding = helpers.get_embedding(bedrock, text[0])
                embedding_list.append(embedding)
                distance = helpers.calculate_dot_product(prompt_embedding, embedding)
                txt_array.append(text[0])
                distance_array.append(distance)
                prompt_display.append(prompt[0])
                
            df = pd.DataFrame({'Prompt':prompt_display,'Text':txt_array, 'Dot Product':distance_array})

        
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
    with st.expander("See Code"):
        st.code(code_data, language="python")

    txt_array.append(prompt[0])    
    
    if prompt_embedding:
        embedding_list.append(prompt_embedding)
        df_embeddings = pd.DataFrame({'Text':txt_array, 'Embeddings':embedding_list})
        
        df_vectors = pd.DataFrame(np.column_stack(list(zip(*df_embeddings[['Embeddings']].values))))
        df_vectors.index = df_embeddings['Text']
        st.subheader("Embeddings")
        st.dataframe(df_vectors)
        
        
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        pca = PCA(n_components=2, svd_solver='auto')
        pca_result = pca.fit_transform(df_vectors.values)
        #display(df)fig = plt.figure(figsize=(14, 8))
        x = list(pca_result[:,0])
        y = list(pca_result[:,1])
        # x and y given as array_like objects
        
        import plotly.express as px
        fig = px.scatter(df_embeddings, x=x, y=y, color=df_vectors.index, hover_name=df_vectors.index)
        fig.update_traces(textfont_size=10)
        # fig.show()
        st.subheader("Vector Space")
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

    if df is not None:
        st.subheader("Dot Product between :orange[Prompt] and :orange[Text]")
        st.table(df)