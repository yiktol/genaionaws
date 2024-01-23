import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Text generation")

t = '''
### Text generation

Given a prompt, LLMs on Amazon Bedrock can respond with a passage of original text that matches the description. Here is one example:
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
    

st.write(":orange[Template:]")
template = '''
Please write a {Text_Category} in the voice of {Role}.
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(category,role):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Text_Category","Role"], template=template)
    prompt_query = prompt.format(
            Text_Category=category,
            Role=role)
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    category = "email"
    role = "friend"
    text_prompt = st.text_area(":orange[User Prompt:]", 
                              height = 50,
                              disabled = False,
                              value = (f"Please write an {category} in the voice of a {role} congratulating someone on a new job."))
    submitted = st.form_submit_button("Submit")
if text_prompt and submitted:
    call_llm(category,role)
