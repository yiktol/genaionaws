
import streamlit as st
import json
import boto3
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)

modelId = 'ai21.j2-mid'
prompt = """Review: 
Extremely old cabinets, phone was half broken and full of dust. Bathroom door was broken, bathroom floor was dirty and yellow. Bathroom tiles were falling off. Asked to change my room and the next room was in the same conditions. The most out of date and least maintained hotel i ever been on.
Extracted sentiment:
{“Cleaning”: “Negative”, “Hotel Facilities”: “Negative”, “Room Quality”: “Negative”}
## 
Review: 
Great experience for two teenagers. We would book again. Location good. 
Extracted sentiment:
{“Location”: “Positive”}
## 
Review: 
Pool area was definitely a little run down and did not look like the pics online at all. Bathroom in the double room was kind of dumpy.
Extracted sentiment:
{“Pool”: “Negative”, “Room Quality”: “Negative”}
## 
Review: 
Roof top's view is gorgeous and the lounge area is comfortable. The staff is very courteous and the location is great. The hotel is outdated and the shower need to be clean better. The air condition runs all the time and cannot be control by the temperature control setting. 
Extracted sentiment:
{“Cleaning”: “Negative”, “AC”: “Negative”, “Room Quality”: “Negative”, “Service”: “Positive”, “View”: “Positive”, “Hotel Facilities”: “Positive”}
## 
Review: 
First I was placed near the elevator where it was noises, the TV is not updated, the toilet was coming on and off. There was no temp control and my shower curtain smelled moldy. Not sure what happened to this place but only thing was a great location.
Extracted sentiment:
{“Cleaning”: “Negative”, “AC”: “Negative”, “Room Quality”: “Negative”, “Location”: “Positive”}
## 
Review: 
This is a very well located hotel and it's nice and clean. Would stay here again. 
Extracted sentiment:
{“Cleaning”: “Positive”, “Location”: “Positive”}
## 
Review: 
Hotel is horrendous. The room was dark and dirty. No toilet paper in the bathroom. Came here on a work trip and had zero access to WiFi even though their hotel claims to have WiFi service. I will NEVER return.
Extracted sentiment:
{“Cleaning”: “Negative”, “WiFi”: “Negative”, “Room Quality”: “Negative”, “Service”: “Negative”}
## 
Review: 
The rooms are small but clean and comfortable. The front desk was really busy and the lines for check-in were very long but the staff handled each person in a professional and very pleasant way. We will stay there again. 
Extracted sentiment:
{“Cleaning”: “Positive”, “Service”: “Positive”}
## 
Review: 
The stay was very nice would stay again. The pool closes at 7 pm and doesn't open till 11am m. That sucked. Also our wifi went out the entire last day we were there. Thats sucked too. Overall was a nice enough stay and I love the location.
Extracted sentiment:
"""

code_data = f"""import json
import boto3

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )

modelId = 'ai21.j2-mid'
accept = 'application/json'
contentType = 'application/json'

prompt= \"""{prompt}\"""

input = {{
    'prompt':prompt, 
    'maxTokens': 200,
    'temperature': 0.3,
    'topP': 1.0,
    'stopSequences': [],
    'countPenalty': {{'scale': 0}},
    'presencePenalty': {{'scale': 0}},
    'frequencyPenalty': {{'scale': 0}}
        }}
body=json.dumps(input)
response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
response_body = json.loads(response.get('body').read())
completions = response_body['completions']
for part in completions:
    print(part['data']['text'])

"""

with text:

    # st.title("Extract Action Items")
    st.header("Extract Topics and Sentiments from Reviews")
    st.write("In this example, we want to extract topics and sentiments from reviews. Each entry in the prompt has a review and an JSON string with extracted topics and sentiments. The last entry in the promot has only the review. The expectation is to extract the topic and sentiment from the last entry.")
    with st.form('form1'):
        prompt = st.text_area(":orange[Article]", prompt, height=500)
        submit = st.form_submit_button("Extract Topics and Sentiments",type='primary')
        
    if submit:
        output = invoke_model(client=bedrock_runtime_client(), prompt=prompt, model=modelId)
        #print(output)
        st.info(output)
    
with code:
    

    st.code(code_data, language="python")

