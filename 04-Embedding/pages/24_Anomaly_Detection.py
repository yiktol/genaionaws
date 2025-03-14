import pandas as pd
import streamlit as st
import utils.helpers as helpers


helpers.set_page_config()
bedrock = helpers.bedrock_runtime_client()

text, code = st.columns([0.6, 0.4])

with text:

    st.header("Anomaly Detection")
    st.markdown("""Now let's look into a slightly different name list. Most of the names in the list are physicists, with only a few that are not physicists. How do we identify the names that are not physicists?\n
_names = ['Albert Einstein', 'Isaac Newton', 'Stephen Hawking', 'Galileo Galilei', 'Niels Bohr', 'Werner Heisenberg', 'Marie Curie', 'Ernest Rutherford', 'Michael Faraday', 'Richard Feynman', 'Lady Gaga', 'Erwin Schrödinger', 'Max Planck', 'Enrico Fermi', 'Taylor Swift', 'Lord Kelvin']_\n
An entry level anomaly detection can be achieved by the following steps:
- Treat all entries as one whole body and calculate the center of mass of the body.
- For all entries, calculate their distance from the center of mass.
- Sort the distances in reverse order.
- The entry that is furthest away from the center of mass is considered an outlier.\n
Here is the sample code to identify two outliers from a name list:""")

    with st.form("myform"):
        names = st.multiselect(":orange[Select Names:]", 
                            ['Albert Einstein', 'Isaac Newton', 'Stephen Hawking', 
         'Galileo Galilei', 'Niels Bohr', 'Werner Heisenberg', 
         'Marie Curie', 'Ernest Rutherford', 'Michael Faraday', 
         'Richard Feynman', 'Lady Gaga', 'Erwin Schrödinger', 
         'Max Planck', 'Enrico Fermi', 'Taylor Swift', 'Lord Kelvin'],['Albert Einstein', 'Isaac Newton', 'Stephen Hawking', 
         'Galileo Galilei', 'Niels Bohr', 'Werner Heisenberg', 
         'Marie Curie', 'Ernest Rutherford', 'Michael Faraday', 
         'Richard Feynman', 'Lady Gaga', 'Erwin Schrödinger', 
         'Max Planck', 'Enrico Fermi', 'Taylor Swift', 'Lord Kelvin']),
        outlier_by_number = st.slider("Outlier by Number", min_value=1, max_value=10, value=2,step = 1)
        submit = st.form_submit_button("Get Outliers by Number",type="secondary")
        outlier_by_percent=st.slider("Outlier by Percent", min_value=1, max_value=100, value=10,step = 1)
        submit2 = st.form_submit_button("Get Outliers by Percent",type="secondary")
        outlier_by_distance=st.slider("Outlier by Distance", min_value=1, max_value=100, value=60,step = 1)
        submit3 = st.form_submit_button("Get Outliers by Distance",type="secondary")
        
    if outlier_by_number and submit:
        with st.spinner("Looking for Outliers..."):
            dataset = []
            for name in names[0]:
                embedding = helpers.get_embedding(bedrock, name)
                dataset.append({'name': name, 'embedding': embedding})
            outliers = helpers.find_outliers_by_count(dataset, outlier_by_number)
            #print(outliers)
            outlier_names = []
            for item in outliers[0]:
                #print(item['name'])
                outlier_names.append(item['name'])
            df = pd.DataFrame({'Outliers':outlier_names})
            st.write("Answer:")
            st.table(df)            
    if outlier_by_percent and submit2:
        with st.spinner("Looking for Outliers..."):
            dataset = []
            for name in names[0]:
                embedding = helpers.get_embedding(bedrock, name)
                dataset.append({'name': name, 'embedding': embedding})
            entries = helpers.find_outliers_by_percentage(dataset, outlier_by_percent)
            outlier_names = []
            for item in entries:
                #print(item['name'])
                outlier_names.append(item['name'])
            df = pd.DataFrame({'Outliers':outlier_names})
            st.write("Answer:")
            st.table(df)  
    if outlier_by_distance and submit3:
        with st.spinner("Looking for Outliers..."):
            dataset = []
            for name in names[0]:
                embedding = helpers.get_embedding(bedrock, name)
                dataset.append({'name': name, 'embedding': embedding})
            entries = helpers.find_outliers_by_distance(dataset, outlier_by_distance)
            outlier_names = []
            for item in entries:
                #print(item['name'])
                outlier_names.append(item['name'])
            df = pd.DataFrame({'Outliers':outlier_names})
            st.write("Answer:")
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
    
def find_outliers_by_count(dataset, count):
    # find the center of mass
    embeddings = []
    for item in dataset:
        embeddings.append(item['embedding'])
    center = np.mean(embeddings, axis=0)
    # calculate distance from center
    for item in dataset:
        item['distance'] = calculate_distance(item['embedding'], center)
    # sort the distances in reverse order
    dataset.sort(key=lambda x: x['distance'], reverse=True)
    # return N outliers
    return dataset[0:count]

# main function
region_name = 'us-east-1'
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)
names = ['Albert Einstein', 'Isaac Newton', 'Stephen Hawking', 
         'Galileo Galilei', 'Niels Bohr', 'Werner Heisenberg', 
         'Marie Curie', 'Ernest Rutherford', 'Michael Faraday', 
         'Richard Feynman', 'Lady Gaga', 'Erwin Schrödinger', 
         'Max Planck', 'Enrico Fermi', 'Taylor Swift', 'Lord Kelvin']
dataset = []
for name in names:
    embedding = get_embedding(bedrock, name)
    dataset.append({'name': name, 'embedding': embedding})
outliers = find_outliers_by_count(dataset, 2)
for item in outliers:
    print(item['name'])
        """
    st.subheader("Code")
    st.code(code_data, language="python")



            

