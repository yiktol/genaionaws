import streamlit as st
import pandas as pd
import numpy as np
import helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()


animals = ["Coyotes","Wolves","Foxes","Ducks","Eagles","Owls",
           "Vultures","Woodpeckers","Cheetahs","Jaguars","Lions","Tigers",
           "Gorillas","Monkeys","Horses","Elephants","Rabbits"
           ]


st.header("Word Embeddings")
st.write("After the step of tokenization, the transformer model then encodes these tokens into n-dimensional vectors.")

with st.form("myform"):
    selected_animals = st.multiselect("What your favorite anilmals:",animals,animals)
    submit = st.form_submit_button("Generate Embeddings",type="primary")

txt_array=[]
distance_array=[]
prompt_display = []
embedding_list =[]

if submit:
    with st.spinner("Generating..."):
        for animal in selected_animals:
            embedding = helpers.get_embedding(bedrock, animal)
            embedding_list.append(embedding)
            txt_array.append(animal)


    df_embeddings = pd.DataFrame({'Text':txt_array, 'Embeddings':embedding_list})
    
    df_vectors = pd.DataFrame(np.column_stack(list(zip(*df_embeddings[['Embeddings']].values))))
    df_vectors.index = df_embeddings['Text']
    st.subheader("Vectors")
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


