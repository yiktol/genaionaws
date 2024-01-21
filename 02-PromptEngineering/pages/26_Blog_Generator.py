import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config

set_page_config()

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        provider = st.selectbox('Provider',('Antropic','Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        # temperature = st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        # top_k=st.number_input('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        # top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        # max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 4096, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
    
    
st.title("🦜🔗 Blog Outline Generator")
st.markdown("**Template:** \n\"As an experienced data scientist and technical writer, generate an outline for a blog about :orange[{topic}].\"")

#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)


def blog_outline(topic):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    template = "As an experienced data scientist and technical writer, generate an outline for a blog about {topic}."
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_input("Enter a topic:", "")
    submitted = st.form_submit_button("Submit")
if topic_text and submitted:
    blog_outline(topic_text)
