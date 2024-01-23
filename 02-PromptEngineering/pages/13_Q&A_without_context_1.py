import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
    
row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Question-answer")

t = '''
### Question-answer, without context

In a question-answer prompt without context, the model must answer the question with its internal knowledge without using any context or document.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

st.write(":orange[Template:]")
template = '''
{Question}
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(question):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Question"], template=template)
    prompt_query = prompt.format(
            Question=question
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    question = "What is Robert Frost's \"Stopping by the woods on a snowy evening\" about metaphorically?"
    text_prompt= st.text_area(":orange[User Prompt:]", 
                              height = 50,
                              disabled = False,
                              value = question)
    submitted = st.form_submit_button("Submit")
if question and submitted:
    call_llm(text_prompt)
