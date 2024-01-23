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

In the following example, a user prompts the model to take on the role of a specific person when generating the text. Notice how the signature reflects the role the model is taking on in the response.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
        
st.write(":orange[Template:]")
template = '''
{Role_assumption} {Task_description}.
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(task,role):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Role_assumption","Task_description"], template=template)
    prompt_query = prompt.format(
            Role_assumption=role,
            Task_description=task)
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    task = "Help me write a note expressing my gratitude to my parents for taking my son (their grandson) to the zoo. I miss my parents so much."
    role = "My name is Jack."
    text_prompt = st.text_area(":orange[User Prompt:]", 
                              height = 50,
                              disabled = False,
                              value = (f"{role} {task}"))
    submitted = st.form_submit_button("Submit")
if text_prompt and submitted:
    call_llm(task,role)
