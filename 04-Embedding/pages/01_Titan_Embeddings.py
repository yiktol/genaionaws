import boto3
import json
import streamlit as st
import numpy as np
import pandas as pd
from helpers import set_page_config

set_page_config()

#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

model_id = 'amazon.titan-embed-g1-text-02' #look for embeddings in the modelID
accept = 'application/json' 
content_type = 'application/json'

st.title("Vertorize using Titan Embeddings")
st.write("""The new Titan Embeddings G1 - Text v1.2 can intake up to 8k tokens and outputs a vector of 1536 dimensions.\
The model also works in 25+ different languages. The model is optimized for text retrieval tasks but can also perform\
additional tasks such as semantic similarity and clustering. \
Titan Embeddings G1 - Text v1.2 also supports long documents, however, for retrieval tasks it is recommended to segment \
documents into logical segments.""")

# Define prompt and model parameters
with st.form("myform"):
    prompt_data = st.text_input(
        ":orange[Enter a prompt to vectorize]", 
        placeholder="Write me a poem about apples",
        value="Write me a poem about apples")
    submitted = st.form_submit_button("Vectorize",type="primary")

if prompt_data and submitted:
    body = json.dumps({
        "inputText": prompt_data,
    })


    # Invoke model 
    response = bedrock_runtime.invoke_model(
        body=body, 
        modelId=model_id, 
        accept=accept, 
        contentType=content_type
    )

    # Print response
    response_body = json.loads(response['body'].read())
    embedding = response_body.get('embedding')

    numbers = (np.array(embedding).reshape(128,12))
    df = pd.DataFrame(numbers, columns=("col %d" % i for i in range(12)))
    st.dataframe(df,use_container_width=True,height=500)