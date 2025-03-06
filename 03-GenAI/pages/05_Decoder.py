import streamlit as st
import pandas as pd
import numpy as np
import helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()


prompt_embedding = []

text, code = st.columns([0.5, 0.5])


animals = ["cat","bird","bear","human","tiger","lion","leopard","dog","puppy","panda","dolphin","squirrel","mouse","hamster",]
planets = ["mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"]
colors = ["orange","yellow","green","blue","purple","pink","brown","gray","black","white"]


dataset = [
    {"id": 1, "text": "A puppy is to dog as kitten is to  .", "options": animals},
    {"id": 2, "text": "Men are from Mars, Women are from  .", "options": planets},
    {"id": 3, "text": "Roses are red Violets are  .", "options": colors}
    
]



distance_array=[]
prompt_display = []
with text:
    st.header("Decoder")
    
    def token_form(item_num):
        with st.form(f"myform-{item_num}"):
            prompt = st.text_area(":orange[Enter your prompt here:]", height = 68, value=dataset[item_num]['text']),
            options=st.multiselect('Options',dataset[item_num]['options'],dataset[item_num]['options'])
            submit = st.form_submit_button("Decode",type="primary")
            # print(prompt[0])
            
        return submit, prompt[0], options


    def show(submit, prompt, options, item_num):
        txt_array=[]
        embedding_list =[]
        prompt_embedding = []
        if prompt and submit:
            with st.spinner("Calculating Distance..."):
                prompt_embedding = helpers.get_embedding(bedrock, prompt)
                
                for text in options:
                    embedding = helpers.get_embedding(bedrock, text)
                    embedding_list.append(embedding)
                    distance = helpers.calculate_distance(prompt_embedding, embedding)
                    txt_array.append(text)
                    distance_array.append(distance)
                    prompt_display.append(prompt)
        
                df = pd.DataFrame({'Prompt':prompt_display,'Text':txt_array, 'Distance':distance_array})
                st.write("Distance between :orange[Prompt] and :orange[Text]")
                st.table(df)            
                
        return prompt_embedding, txt_array, embedding_list
                
    tabs = st.tabs([f"Item {i+1}" for i in range(len(dataset))])

    for i, tab in enumerate(tabs):
        with tab:
            submit, prompt, options = token_form(i)
            pe, ta, el = show(submit, prompt, options, i)
            # print(pe, ta)


        # st.dataframe(df_embeddings)
     
     
with code:   
    
    def output(prompt_embedding, txt_array, embedding_list, prompt):
    
        txt_array.append(prompt)    
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
                
    output(pe, ta, el, prompt)
    


