from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()



row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Good vs Bad Prompt")

t = '''
### Provide simple, clear, and complete instructions

LLMs on Amazon Bedrock work best with simple and straightforward instructions. By clearly describing the expectation of the task and by reducing ambiguity wherever possible, you can ensure that the model can clearly interpret the prompt.

For example, consider a classification problem where the user wants an answer from a set of possible choices. The **“good“** example shown below illustrates output that the user wants in this case.

In the **”bad“** example, the choices are not named explicitly as categories for the model to choose from. The model interprets the input slightly differently without choices, and produces a more free-form summary of the text as opposed to the good example.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','Antropic'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()


textgen_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)


good_prompt = """The most common cause of color blindness is an inherited problem or variation in the functionality \
of one or more of the three classes of cone cells in the retina, which mediate color vision.\n
What is the above text about?
a) biology
b) history
c) geology """

bad_prompt = """Classify the following text.\n
\"The most common cause of color blindness is an inherited problem or variation in the functionality \
of one or more of the three classes of cone cells in the retina, which mediate color vision.\""""

prompt_type = st.selectbox(
    ":orange[Select Prompt Type:]",("Good Prompt","Bad Prompt"))

if prompt_type == "Good Prompt":
    with st.form("myform1"):
        prompt_data = st.text_area(
        ":orange[Good Prompt:]",
        height = 200,
        value = good_prompt
        )
        submit = st.form_submit_button("Submit")
else:
    with st.form("myform2"):
        prompt_data = st.text_area(
        ":orange[Bad Prompt:]",
        height = 120,
        value = bad_prompt
        )

        submit = st.form_submit_button("Submit")

if prompt_data and submit:

    response = textgen_llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.write(response)
  