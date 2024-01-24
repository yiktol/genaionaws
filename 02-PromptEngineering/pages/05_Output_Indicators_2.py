from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Output indicators")

t = '''
### Output indicators

Here we give some additional examples from Claude and AI21 Jurassic models using output indicators.

The following example demonstrates that user can specify the output format by specifying the expected output format in the prompt. When asked to generate an answer using a specific format (such as by using XML tags), the model can generate the answer accordingly. Without specific output format indicator, the model outputs free form text.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Anthropic','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

textgen_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)

prompt = """Human: Extract names and years: the term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence. The synonym self-teaching computers was also used in this time period."""

prompt_type = st.selectbox(
    ":orange[Select Prompt Type]",("Prompt with clear output constraints indicator","Prompt without clear output specifications"))

if prompt_type == "Prompt with clear output constraints indicator":
    with st.form("myform1"):
        prompt_data = st.text_area(
        ":orange[User Prompt:]",
        height = 200,
        value = f"{prompt}\n\nPlease generate answer in <name></name> and <year></year> tags.\n\nAssistant:"
        )
        submit = st.form_submit_button("Submit")
else:
    with st.form("myform2"):
        prompt_data = st.text_area(
        ":orange[User Prompt:]",
        height = 200,
        value = f"{prompt}\n\nAssistant:"
        )
        submit = st.form_submit_button("Submit")

if prompt_data and submit:

    response = textgen_llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.write(response)
  