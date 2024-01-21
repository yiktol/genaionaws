import boto3
from langchain.llms.bedrock import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

st.title("Answer step by step")

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        provider = st.selectbox('Provider',('AI21','Amazon','Antropic'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

t = '''
### Complex tasks: build toward the answer step by step
LLM models can provide clear steps for certain tasks, and including the phrase Think step-by-step to come up with the right answer can help produce the appropriate output.
'''

st.markdown(t)

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

textgen_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)

prompt = "At a Halloween party, Jack gets 15 candies. Jack eats 5 candies. He wants to give each friend 5 candies. How many friends can receive candies?\n\nThink step-by-step to come up with the right answer."


with st.form("myform1"):
    prompt_data = st.text_area(
    ":orange[User Prompt:]",
    height = 120,
    value = f"{prompt}"
    )
    submit = st.form_submit_button("Submit")


if prompt_data and submit:

    response = textgen_llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.write(response)
  