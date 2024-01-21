import boto3
from langchain.llms.bedrock import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

st.title("Good vs Bad Prompt")

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','Antropic'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 


t = '''
### Provide simple, clear, and complete instructions

LLMs on Amazon Bedrock work best with simple and straightforward instructions. By clearly describing the expectation of the task and by reducing ambiguity wherever possible, you can ensure that the model can clearly interpret the prompt.

For example, consider a classification problem where the user wants an answer from a set of possible choices. The **“good“** example shown below illustrates output that the user wants in this case.

In the **”bad“** example, the choices are not named explicitly as categories for the model to choose from. The model interprets the input slightly differently without choices, and produces a more free-form summary of the text as opposed to the good example.
'''

st.markdown(t)

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()


textgen_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)

prompt_type = st.selectbox(
    ":orange[Select Prompt Type:]",("Good Prompt","Bad Prompt"))

if prompt_type == "Good Prompt":
    with st.form("myform1"):
        prompt_data = st.text_area(
        ":orange[Good Prompt:]",
        height = 200,
        value = """The most common cause of color blindness is an inherited problem or variation in the functionality of one or more of the three classes of cone cells in the retina, which mediate color vision.\n\nWhat is the above text about? \na) biology \nb) history \nc) geology """
        )
        submit = st.form_submit_button("Submit")
else:
    with st.form("myform2"):
        prompt_data = st.text_area(
        ":orange[Bad Prompt:]",
        height = 100,
        value = """Classify the following text. "The most common cause of color blindness is an inherited problem or variation in the functionality of one or more of the three classes of cone cells in the retina, which mediate color vision."""
        )

        submit = st.form_submit_button("Submit")

if prompt_data and submit:

    response = textgen_llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.write(response)
  