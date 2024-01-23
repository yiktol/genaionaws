import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Code generation")

t = '''
### Code generation

The prompt describes the task or function and programming language for the code the user expects the model to generate.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

st.write(":orange[Template:]")
template = '''
Write a function in {programming_language} to {task_description}
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(task,language):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["programming_language","Task_description"], template=template)
    prompt_query = prompt.format(
            programming_language=language,
            task_description=task)
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)

task = st.selectbox(':orange[Task]',("download a s3 file to local disk","quickly approximates the square root of a number."))
language = st.selectbox(":orange[Language]",("python","javascript","ruby","c#","nodejs"))
with st.form("myform"):
    text_prompt = st.text_area(":orange[User Prompt:]", 
                            height = 50,
                            disabled = True,
                            value = (f"Write a function in {language} to {task}."))        
        
        
    submitted = st.form_submit_button("Submit")
    
if  text_prompt and submitted:
    call_llm(task,language)
