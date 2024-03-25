import streamlit as st
import pandas as pd
import numpy as np
import utils.helpers as helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()

st.header("Similarity Metrics")

left_co, cent_co,last_co = st.columns([0.2,0.2,0.3])
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
        submit = st.form_submit_button("Compare",type="primary")

    txt_array=[]
    distance_array=[]
    dot_product_array=[]
    similarity_array=[]
    prompt_display = []
    embedding_list =[]
    prompt_embedding = []

    #print(text1)
    if prompt and submit:
        with st.spinner("Comparing.."):
            prompt_embedding = helpers.get_embedding(bedrock, prompt[0])
            
            texts = [text1, text2, text3, text4, text5]
            for text in texts:
                embedding = helpers.get_embedding(bedrock, text[0])
                embedding_list.append(embedding)
                similarity = helpers.calculate_cosine_similarity(prompt_embedding, embedding)
                distance = helpers.calculate_distance(prompt_embedding, embedding)
                dot_product = helpers.calculate_dot_product(prompt_embedding, embedding)
                txt_array.append(text[0])
                similarity_array.append(similarity)
                distance_array.append(distance)
                dot_product_array.append(dot_product)
                prompt_display.append(prompt[0])
                
                
            df = pd.DataFrame({'Prompt':prompt_display,'Text':txt_array, 'Cosine Similarity':similarity_array, 'Dot Product':dot_product_array, 'Euclidean Distance':distance_array})
            
with code:

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
    st.subheader("Similarity Metrics between :orange[Prompt] and :orange[Text]")
    st.table(df)