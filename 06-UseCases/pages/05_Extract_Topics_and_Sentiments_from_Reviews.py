
import streamlit as st
import textwrap
from helpers import bedrock_runtime_client, set_page_config, invoke_model,getmodelId


set_page_config()


text, code = st.columns([0.6,0.4])

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



with code:
    with st.container(border=True):
        provider = st.selectbox('Provider',('Amazon','Anthropic','AI21','Cohere','Meta'))
        model_id=st.text_input('model_id',getmodelId(provider))
    
    with st.form(key ='form2'):
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
        topP=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
        maxTokens=st.number_input('maxTokens',min_value = 50, max_value = 4096, value = 1024, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters')    

    code_data = f"""import json
import boto3

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )

modelId = 'ai21.j2-mid'
accept = 'application/json'
contentType = 'application/json'

prompt= \"{textwrap.shorten(prompt,width=50,placeholder='...')}\"

input = {{
    'prompt':prompt, 
    'maxTokens': {maxTokens},
    'temperature': {temperature},
    'topP': {topP},
    'stopSequences': [],
    'countPenalty': {{'scale': 0}},
    'presencePenalty': {{'scale': 0}},
    'frequencyPenalty': {{'scale': 0}}
        }}
        
response = bedrock.invoke_model(
    body=json.dumps(input),
    modelId=modelId, 
    accept=accept,
    contentType=contentType
    )
    
response_body = json.loads(response.get('body').read())
completions = response_body['completions']

for part in completions:
    print(part['data']['text'])

"""

    with st.expander("Show Code"):
        st.code(code_data, language="python")


with text:

    # st.title("Extract Action Items")
    st.header("Extract Topics and Sentiments from Reviews")
    st.write("In this example, we want to extract topics and sentiments from reviews. Each entry in the prompt has a review and an JSON string with extracted topics and sentiments. The last entry in the promot has only the review. The expectation is to extract the topic and sentiment from the last entry.")
    with st.form('form1'):
        prompt = st.text_area(":orange[Article]", prompt, height=500)
        submit = st.form_submit_button("Extract Topics and Sentiments",type='primary')
        
    if submit:
        with st.spinner("Thinking..."):
            output = invoke_model(client=bedrock_runtime_client(), 
                                prompt=prompt, 
                                model=model_id,
                                temperature=temperature,
                                top_p=topP,
                                max_tokens=maxTokens,)
            #print(output)
            st.write("Answer:")
            st.info(output)
        