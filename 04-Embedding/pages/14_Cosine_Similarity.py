import streamlit as st
import pandas as pd
import numpy as np
import utils.helpers as helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()
st.header("Cosine Similarity")
col1, col2, col3 = st.columns([0.2, 0.2,0.6])
col1.image("dot-product-a-cos.svg")
col2.latex(r'\cos \theta = \frac{a.b}  {|a| x |b|}') 

st.write("""Cosine Similarity is another commonly used method to measure similarity.\
In linear algebra, cosine similarity is the cosine of the angle between two vectors.\
That is, it is the dot product of the vectors divided by the product of their lengths. \
Similar to Euclidean distance, cosine similarity measures how similar two pieces of text are likely to be in terms of their subject matter,\
which is independent of the length of the texts. When cosine similarity equals to 1,\
it means two pieces of text have identical meanings in natural language.
""")

df = None

text, code = st.columns([0.4, 0.6])

with text:


    with st.form("myform"):
        prompt = st.text_area(":orange[Enter your prompt here:]", height = 50, value="Hello"),
        text1=st.text_area('Text1',value="Hi"),
        text2=st.text_area('Text2',value="Good Day"),
        text3=st.text_area('Text3',value="How are you"),
        text4=st.text_area('Text4',value="Good Morning"),
        text5=st.text_area('Text5',value="Goodbye"),
        submit = st.form_submit_button("Check Similarity",type="primary")

    txt_array=[]
    distance_array=[]
    prompt_display = []
    embedding_list =[]
    prompt_embedding = []

    #print(text1)
    if prompt and submit:
        with st.spinner("Calculating Cosine Similarity..."):
            prompt_embedding = helpers.get_embedding(bedrock, prompt[0])
            
            texts = [text1, text2, text3, text4, text5]
            for text in texts:
                embedding = helpers.get_embedding(bedrock, text[0])
                embedding_list.append(embedding)
                distance = helpers.calculate_cosine_similarity(prompt_embedding, embedding)
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
        st.subheader("Cosine Similarity between :orange[Prompt] and :orange[Text]")
        st.table(df)