import streamlit as st
import pandas as pd
import numpy as np
import helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()


prompt_embedding = []

text, code = st.columns([0.4, 0.6])


with text:
    st.header("Decoder")
  
    with st.form("myform"):
        prompt = st.text_area(":orange[Enter your prompt here:]", height = 30, value="A puppy is to dog as kitten is to"),
        text1=st.text_input('Text1',value="cat"),
        text2=st.text_input('Text2',value="bird"),
        text3=st.text_input('Text3',value="bear"),
        text4=st.text_input('Text4',value="human"),
        text5=st.text_input('Text5',value="tiger"),
        submit = st.form_submit_button("Decode",type="primary")

    txt_array=[]
    distance_array=[]
    prompt_display = []
    embedding_list =[]

    if prompt and submit:
        with st.spinner("Calculating Distance..."):
            prompt_embedding = helpers.get_embedding(bedrock, prompt[0])

            texts = [text1, text2, text3, text4, text5]
            for text in texts:
                embedding = helpers.get_embedding(bedrock, text[0])
                embedding_list.append(embedding)
                distance = helpers.calculate_distance(prompt_embedding, embedding)
                txt_array.append(text[0])
                distance_array.append(distance)
                prompt_display.append(prompt[0])
    
            df = pd.DataFrame({'Prompt':prompt_display,'Text':txt_array, 'Distance':distance_array})
            st.write("Distance between :orange[Prompt] and :orange[Text]")
            st.table(df)            
            
            


        # st.dataframe(df_embeddings)
     
     
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


