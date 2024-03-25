import streamlit as st
import numpy as np
import pandas as pd
import utils.helpers as helpers

helpers.set_page_config()

bedrock = helpers.bedrock_runtime_client()

st.header("Euclidean Distance")
col1, col2, col3 = st.columns([0.3, 0.3,0.4])
col1.image("dist-2-points-a.svg")
col2.latex(r'd(p,q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}')

st.write("""When you use vectors to represent text, you can calculate the Euclidean distance between two pieces of text. \
You might wonder why people care about “the distance between two pieces of text”. What does it really mean? \

The Euclidean distance between two pieces of text indicates how far away they are from each other. \
In human readable terms, it means how similar they are in natural language. \
If the distance is very small, the two pieces of text convey a similar message. \
If the distance is very big, the two pieces of text convey a different message.""")

df = None

text, code = st.columns([0.4, 0.6])

with text:

    with st.form("myform"):
        prompt = st.text_area(":orange[Enter your prompt here:]", height = 30, value="Hello"),
        text1=st.text_area('Text1',value="Hi"),
        text2=st.text_area('Text2',value="Good Day"),
        text3=st.text_area('Text3',value="How are you"),
        text4=st.text_area('Text4',value="Good Morning"),
        text5=st.text_area('Text5',value="Goodbye"),
        submit = st.form_submit_button("Check Distance",type="primary")

    txt_array=[]
    distance_array=[]
    prompt_display = []
    embedding_list =[]
    prompt_embedding = []

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



with code:
    
    code_data=f"""import json
import boto3
import math

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

def calculate_distance(v1, v2):
    distance = math.dist(v1, v2)
    return distance

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
    distance = calculate_distance(hello, embedding)
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
        st.subheader("Distance between :orange[Prompt] and :orange[Text]")
        st.table(df)
