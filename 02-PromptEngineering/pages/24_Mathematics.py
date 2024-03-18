import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
import utils.helpers as helpers

helpers.set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Mathematics")

t = '''
### Mathematics

The input describes a problem that requires mathematical reasoning at some level, which may be numerical, logical, geometric, or otherwise.\
For such problem, it's helpful to ask the model to work through the problem in a piecemeal manner by adding phrases to instructions such as *Let\'s think step by step* or *Think step by step to come up with the right answer*.
'''

row1_col1.markdown(t)
with row1_col2:
    with st.container(border=True):
        provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
        model_id=st.text_input('model_id',helpers.getmodelId(provider))

template1 = '''
{Math_problem_description} Let\'s think step by step.
'''
template2 = '''
{Math_problem_description} Think step by step to come up with the right answer.
'''
template = st.selectbox(":orange[Template:] ",(template1,template2))

# st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = helpers.bedrock_runtime_client()

def call_llm(context):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=helpers.getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Math_problem_description"], template=template)
    prompt_query = prompt.format(
            Math_problem_description=context)
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


context = st.selectbox(":orange[Math Problem Description:]",
                        ("A triangle has two angles of 70 degrees and 50 degrees. What is the third angle in degrees?",
                        "Lucy has 12 colorful marbles, and she wants to share them equally with her 4 friends. How many marbles will each friend receive?"))

with st.form("myform"):


    text_prompt = st.text_area(":orange[User Prompt:]", 
                            height = 50,
                            disabled = True,
                            value = f"{context} {template.replace('{Math_problem_description} ','')}"
                            )        
        
        
    submitted = st.form_submit_button("Submit",type="primary")
if  text_prompt and submitted:
    call_llm(context)
