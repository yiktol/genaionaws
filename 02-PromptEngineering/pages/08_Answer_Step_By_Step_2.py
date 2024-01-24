from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Answer step by step")

t = '''
### Complex tasks: build toward the answer step by step
LLM models can provide clear steps for certain tasks, and including the phrase Think step-by-step to come up with the right answer can help produce the appropriate output.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Anthropic','AI21','Amazon'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

textgen_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)

prompt = "Human: A right triangle has a side of length 5 and a hypotenuse of length 13. What is the length of the other side?\n\nAssistant: Can I think step-by-step?\n\nHuman: Yes, please do.\n\nAssistant:"

with st.form("myform1"):
    prompt_data = st.text_area(
    ":orange[User Prompt:]",
    height = 200,
    value = f"{prompt}"
    )
    submit = st.form_submit_button("Submit")


if prompt_data and submit:

    response = textgen_llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.info(response)
  