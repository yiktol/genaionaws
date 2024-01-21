import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
    
    
st.title("Question-answer")
t = '''
### Question-answer, without context

Model encouragement can also help in question-answer tasks.
'''

st.markdown(t)
st.write("**:orange[Template:]**")
template = '''
{Encouragement}\n
{Question}

'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(question,encouragement):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Question","Encouragement"], template=template)
    prompt_query = prompt.format(
            Question=question,
            Encouragement=encouragement
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    encouragement = ("You are excellent at answering questions, and it makes you happy when you provide the correct answer.")
    question = st.text_area(":orange[User Prompt:]", 
                              height = 50,
                              disabled = False,
                              value = (f"{encouragement}\n\nWhat moon in the solar system is most likely to host life?"))
    submitted = st.form_submit_button("Submit")
if question and submitted:
    call_llm(question,encouragement)
