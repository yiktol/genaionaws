import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
    
row1_col1, row1_col2 = st.columns([0.7,0.3])

row1_col1.title("🦜🔗 Blog Outline Generator")

template = "As an experienced data scientist and technical writer, generate an outline for a blog about {topic}."
row1_col1.text_area(":orange[Template]",
                    value=template,
                    height = 30,
                    disabled = True,)

with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Antropic','Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def blog_outline(topic):
    # Instantiate LLM model
    llm = Bedrock(
        model_id=model_id,
        client=bedrock_runtime,
        model_kwargs=getmodelparams(provider)
        )

    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    print(prompt_query)

    response = llm(prompt_query)

    return st.info(response)


with row1_col1.form("myform"):
    topic_text = st.text_input("Enter a topic:", "")
    submitted = st.form_submit_button("Submit")
if topic_text and submitted:
    blog_outline(topic_text)
