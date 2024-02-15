import streamlit as st
import boto3
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
   
row1_col1, row1_col2 = st.columns([0.7,0.3])

row1_col1.title("✈️ Travel Itinerary Generator")

with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Anthropic','AI21'),disabled=True)
        model_id=st.text_input('model_id',getmodelId(provider))
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_k=st.slider('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        top_p=st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.slider('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 4096, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters') 

template = """Human: As a professional travel agent and a expert tour guide,\n 
Generate a {numdays}-day itinerary for upcoming visit to {city}. {timing}.\n
Assistant:"""
row1_col1.text_area(":orange[Template]",
                    value=template,
                    height = 160,
                    disabled = True,)

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

inference_modifier = {
    "max_tokens_to_sample": max_tokens_to_sample,
    "temperature": temperature,
    "top_k": top_k,
    "top_p": top_p,
    "stop_sequences": ["\n\nHuman"],
}


def itinerary(numdays, city, timing):
    # Instantiate LLM model
    llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=bedrock_runtime,
    model_kwargs=inference_modifier,
)
    prompt= PromptTemplate(input_variables=["numdays","city","timing"],template=template)
    prompt_query = prompt.format(numdays=numdays,city=city,timing=timing)
    response = llm(prompt_query)

    return st.info(response)


with row1_col1.form("myform"):
    numdays = st.text_input("Enter the Number of day(s):", 
                              placeholder="2",
                              value="2")
    city = st.text_input("Enter the City name:",
                         placeholder="Paris",
                         value="Paris")
    check = st.checkbox('Display time and duration')
    submitted = st.form_submit_button("Generate")
    
    
if numdays and city and submitted and check:
    timing = 'Add timing and duration'
    itinerary(numdays, city, timing)
if numdays and city and submitted and not check:
    timing = ''
    itinerary(numdays, city, timing=timing)
    
