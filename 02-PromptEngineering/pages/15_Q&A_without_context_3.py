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

Model encouragement can also help in question-answer tasks.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

st.write(":orange[Template:]")
template = '''
{Encouragement}\n
{Question}\n
{Constraints}
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(question,encouragement,constraints):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Question","Encouragement","Constraints"], template=template)
    prompt_query = prompt.format(
            Question=question,
            Encouragement=encouragement,
            Constraints=constraints
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    encouragement = ("You feel rewarded by helping people learn more about climate change.")
    constraints = ("Assume your audience is composed of high school students.")
    question = st.text_area(":orange[User Prompt:]", 
                              height = 140,
                              disabled = False,
                              value = (f"{encouragement}\n\nCould you please explain what climate change is?\n\n{constraints}"))
    submitted = st.form_submit_button("Submit")
if question and submitted:
    call_llm(question,encouragement,constraints)
