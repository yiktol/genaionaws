
import streamlit as st
import pandas as pd
import textwrap
from helpers import bedrock_runtime_client, set_page_config, invoke_model, search, get_embedding

set_page_config()
bedrock = bedrock_runtime_client()

text, code = st.columns(2)


modelId = 'amazon.titan-text-lite-v1'


with text:

    # st.title("Extract Action Items")
    st.header("Search and Recommendation")
    st.markdown("""Let's assume that you have a collection of documents (the data set). Each document is represented by its embedding. You have been given a query string. The ask is to identify the document that is most relevant to the query string. You can achieve this with the following steps:\n
- Represent the query string with its embedding.
- Calculate the Euclidean distance between the query string and all documents in the data set.
- Sort the distances in ascending order.
- The document with the smallest distance is most relevant to the query string.\n
Here is the sample code to perform search and recommendation:""")

    with st.form("myform"):
        prompt = st.text_area(":orange[Enter your query here:]", height = 50, value='Isaac Newton'),
        t1=st.text_input('Text1',value="The theory of general relativity says that the observed gravitational effect between masses results from their warping of spacetime."),
        t2=st.text_input('Text2',value="Quantum mechanics allows the calculation of properties and behaviour of physical systems. It is typically applied to microscopic systems: molecules, atoms and sub-atomic particles."),
        t3=st.text_input('Text3',value="Wavelet theory is essentially the continuous-time theory that corresponds to dyadic subband transforms — i.e., those where the L (LL) subband is recursively split over and over."),
        t4=st.text_input('Text4',value="Every particle attracts every other particle in the universe with a force that is proportional to the product of their masses and inversely proportional to the square of the distance between their centers."),
        t5=st.text_input('Text5',value="The electromagnetic spectrum is the range of frequencies (the spectrum) of electromagnetic radiation and their respective wavelengths and photon energies."),
        submit = st.form_submit_button("Search and Recommend",type="primary")


    if submit:         
        dataset = [
        {'text': t1, 'embedding': get_embedding(bedrock, t1[0])}, 
        {'text': t2, 'embedding': get_embedding(bedrock, t2[0])}, 
        {'text': t3, 'embedding': get_embedding(bedrock, t3[0])}, 
        {'text': t4, 'embedding': get_embedding(bedrock, t4[0])}, 
        {'text': t5, 'embedding': get_embedding(bedrock, t5[0])}
        ]    
        
        v = get_embedding(bedrock, prompt[0]) 
        result = search(dataset, v)

        st.write("Answer:")
        st.info(result[0][0])
        df = pd.DataFrame({'Text':["t1","t2","t3","t4","t5"], 'Distance':result[1]})
        st.table(df)

with code:

    code_data = f"""import json
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

def search(dataset, v):
    for item in dataset:
        item['distance'] = calculate_distance(item['embedding'], v)
    dataset.sort(key=lambda x: x['distance'])
    return dataset[0]['text']
    
# main function
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )
    
# the data set
t1 = \"{textwrap.shorten(t1[0],width=50,placeholder='...')}\"
t2 = \"{textwrap.shorten(t2[0],width=50,placeholder='...')}\"
t3 = \"{textwrap.shorten(t3[0],width=50,placeholder='...')}\"
t4 = \"{textwrap.shorten(t4[0],width=50,placeholder='...')}\"
t5 = \"{textwrap.shorten(t5[0],width=50,placeholder='...')}\"

dataset = [
    {{'text': t1, 'embedding': get_embedding(bedrock, t1)}}, 
    {{'text': t2, 'embedding': get_embedding(bedrock, t2)}}, 
    {{'text': t3, 'embedding': get_embedding(bedrock, t3)}}, 
    {{'text': t4, 'embedding': get_embedding(bedrock, t4)}}, 
    {{'text': t5, 'embedding': get_embedding(bedrock, t5)}}
    ]
    
# perform a search for Albert Einstein
query = \"{prompt[0]}\"
v = get_embedding(bedrock, query)              
result = search(dataset, v)
print(result)
        """

    st.code(code_data, language="python")



            

