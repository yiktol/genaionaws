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
    
    
st.title("Text classification")
t = '''
### Multiple-choice classification question

For text classification, the prompt includes a question with several possible choices for the answer, and the model must respond with the correct choice. Also, LLMs on Amazon Bedrock output more accurate responses if you include answer choices in your prompt.

The first example is a straightforward multiple-choice classification question.
'''

st.markdown(t)
st.write("**:orange[Template:]**")
template = '''
{context}\n
{question}? Choose from the following:\n
{choice1}\n
{choice2}\n
{choice3}
'''
st.code(template, language='text')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm():
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["context","question","cloice1","choice2","choice3"], template=template)
    prompt_query = prompt.format(
            context="San Francisco, officially the City and County of San Francisco, is the commercial, financial, and cultural center of Northern California. The city proper is the fourth most populous city in California, with 808,437 residents, and the 17th most populous city in the United States as of 2022.",
            question="What is the paragraph above about",
            choice1="A city",
            choice2="A person",
            choice3="An event"
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_area(":orange[User Prompt:]", 
                              height = 210,
                              value = "San Francisco, officially the City and County of San Francisco, is the commercial, financial, and cultural center of Northern California. The city proper is the fourth most populous city in California, with 808,437 residents, and the 17th most populous city in the United States as of 2022.\n\nWhat is the paragraph above about? Choose from the following:\n\nA city\nA person\nAn event")
    submitted = st.form_submit_button("Submit")
if topic_text and submitted:
    call_llm()
