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
    
    
st.title("Text generation")
t = '''
### Text generation

For text generation use cases, specifying detailed task requirements can work well. In the following example, we ask the model to generate response with exclamation points.
'''

st.markdown(t)
st.write("**:orange[Template:]**")
template = '''
{Task_specification}
Please write a {Text_Category} in the voice of {Role}.
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(category,role,task):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Text_Category","Role","Task_specification"], template=template)
    prompt_query = prompt.format(
            Text_Category=category,
            Task_specification=task,
            Role=role)
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    category = "email"
    role = "friend"
    task = "Write text with exclamation points."
    text_prompt = st.text_area(":orange[User Prompt:]", 
                              height = 50,
                              disabled = False,
                              value = (f"{task}\nPlease write an {category} in the voice of a {role} congratulating someone on a new job."))
    submitted = st.form_submit_button("Submit")
if text_prompt and submitted:
    call_llm(category,role,task)
