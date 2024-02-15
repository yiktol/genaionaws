
import streamlit as st
import pandas as pd
from helpers import bedrock_runtime_client, set_page_config, classify, search, get_embedding

set_page_config()
bedrock = bedrock_runtime_client()

text, code = st.columns(2)

with text:

    # st.title("Extract Action Items")
    st.header("Classification")
    st.markdown("""Let’s assume that you have a collection of documents (the data set) and you would like to classify them into N classes (labels). Each class has a label name along with a short description. You can achieve this with the following steps:\n
- Represent each class with the embedding of the label name and description.
- For a given document, calculate the Euclidean distance between the document and all classes.
- Classify the document into the class with the shortest distance.\n
Here is the sample code to classify students into athletics, musician, or magician:""")

    Examples = st.selectbox(
        ":orange[Select Prompt Type:]",("Example1","Example2"))

    if Examples == "Example1":
        with st.form("myform"):
            prompt = st.text_area(":orange[Enter your query here:]", height = 50, value='Ellison sends a spell to prevent Professor Wang from entering the classroom'),
            t1=st.text_area('Class1',value="{'name': 'athletics', 'description': 'all students with a talent in sports'}"),
            t2=st.text_area('Class2',value="{'name': 'musician', 'description': 'all students with a talent in music'}"),
            t3=st.text_area('Class3',value="{'name': 'magician', 'description': 'all students with a talent in witch craft'}"),
            submit = st.form_submit_button("Classify",type="primary")

        classes = [    
            {'name': 'athletics', 'description': 'all students with a talent in sports'}, 
            {'name': 'musician', 'description': 'all students with a talent in music'}, 
            {'name': 'magician', 'description': 'all students with a talent in witch craft'}
        ]
        for item in classes:
            item['embedding'] = get_embedding(bedrock, item['description'])
        # perform a classification
        query = prompt[0]
        v = get_embedding(bedrock, query)      
        if submit:         
            result = classify(classes, v)
            #print(result)
            st.write("Answer:")
            st.info(result[0])
            df = pd.DataFrame({'Class':["class1","class2","class3"], 'Distance':result[1]})
            st.table(df)

    if Examples == "Example2":
        
        with st.form("myform2"):
            prompt = st.text_area(":orange[Enter your query here:]", height = 50, value='Steve helped me solve the problem in just a few minutes. Thank you for the great work!'),
            t1=st.text_area('Class1',value="{'name': 'positive', 'description': 'customer demonstrated positive sentiment in the response.'}"),
            t2=st.text_area('Class2',value="{'name': 'negative', 'description': 'customer demonstrated negative sentiment in the response.'}"),
            submit = st.form_submit_button("Classify",type="primary")
        # the data set
        classes = [    
            {'name': 'positive', 'description': 'customer demonstrated positive sentiment in the response.'}, 
            {'name': 'negative', 'description': 'customer demonstrated negative sentiment in the response.'}
        ]
        for item in classes:
            item['embedding'] = get_embedding(bedrock, item['description'])
        # perform a classification
        query = prompt[0]
        v = get_embedding(bedrock, query)      
        if submit:         
            result = classify(classes, v)
            #print(result)
            st.write("Answer:")
            st.info(result[0])
            df = pd.DataFrame({'Class':["class1","class2"], 'Distance':result[1]})
            st.table(df)

with code:

    code_data = """import json
import boto3
import math

def get_embedding(bedrock, text):
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    input = {
            'inputText': text
        }
    body=json.dumps(input)
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept,contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body['embedding']
    return embedding

def calculate_distance(v1, v2):
    distance = math.dist(v1, v2)
    return distance
    
def classify(classes, v):
    for item in classes:
        item['distance'] = calculate_distance(item['embedding'], v)
    classes.sort(key=lambda x: x['distance'])
    return classes[0]['name']
    
# main function
region_name = 'us-east-1'
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)
# the data set
classes = [    
    {'name': 'athletics', 'description': 'all students with a talent in sports'}, 
    {'name': 'musician', 'description': 'all students with a talent in music'}, 
    {'name': 'magician', 'description': 'all students with a talent in witch craft'}
]
for item in classes:
    item['embedding'] = get_embedding(bedrock, item['description'])
# perform a classification
query = 'Ellison sends a spell to prevent Professor Wang from entering the classroom'
v = get_embedding(bedrock, query)              
result = classify(classes, v)
print(result)
        """

    st.code(code_data, language="python")



            

