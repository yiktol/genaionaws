import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Reasoning/logical thinking")

t = '''
### Reasoning/logical thinking

For complex reasoning tasks or problems that requires logical thinking, we can ask the model to make logical deductions and explain its answers.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','Anthropic'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

template1 = '''Question: {question}\n\nPlease output the answer and then explain your answer:
'''
template2 = '''Human: {question}\n
Please provide the answer and show the reasoning.\n
Assistant:
'''
st.write(":orange[Template:]")
if provider == 'Anthropic':
    template = template2
    st.code(template, language='None')
else:
    template = template1
    st.code(template, language='None')

# st.code(template, language='None')

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
    
    prompt_query = format_prompt()
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)

def format_prompt():
    prompt = PromptTemplate(input_variables=["question"], template=template)
    prompt_query = prompt.format(
            question=context)
    return prompt_query


if provider == 'Anthropic':
    context = "The barber is the \"one who shaves all those, and those only, who do not shave themselves\". Does the barber shave himself? Why is this a paradox?"
else:
    context = "Which word is the odd one out?\nA. accomplished\nB. good\nC. horrible\nD. outstanding"

with st.form("myform"):
    text_prompt = st.text_area(":orange[User Prompt:]", 
                            height = 200,
                            disabled = False,
                            value = format_prompt()
                            )        
        
        
    submitted = st.form_submit_button("Submit")
if  text_prompt and submitted:
    call_llm()
