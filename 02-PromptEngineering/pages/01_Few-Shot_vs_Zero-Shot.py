from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Few-shot vs. zero-shot")

t = '''
### Few-shot prompting vs. zero-shot prompting

It is sometimes useful to provide a few examples to help LLMs better calibrate their output to meet your expectations, also known as few-shot prompting or in-context learning, where a shot corresponds to a paired example input and the desired output. To illustrate, first here is an example of a zero-shot sentiment classification prompt where no example input-output pair is provided in the prompt text:
'''
row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','Anthropic'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

textgen_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)

prompt_type = st.selectbox(
    ":orange[Select Prompt Type:]",("Few-shot","Zero-shot"))

prompt1 = """Tell me the sentiment of the following headline and categorize it as either positive, negative or neutral:\n
New airline between Seattle and San Francisco offers a great opportunity for both passengers and investors."""

prompt2 = """Tell me the sentiment of the following headline and categorize it as either positive, negative or neutral. Here are some examples: \
\n
Research firm fends off allegations of impropriety over new technology.\n
Answer: Negative \
\n
Offshore wind farms continue to thrive as vocal minority in opposition dwindles.\n
Answer: Positive \
\n
Manufacturing plant is the latest target in investigation by state officials.\n
Answer:"""

if prompt_type == "Zero-shot":
    with st.form("myform1"):
        prompt_data = st.text_area(
            ":orange[Zero-shot:]",
            height = 100,
            value = prompt1
            )
        submit = st.form_submit_button("Submit")
else:
    with st.form("myform2"):
        prompt_data = st.text_area(
        ":orange[Few-shot:]",
        height = 350,
        value = prompt2
        )
        submit = st.form_submit_button("Submit")

if prompt_data and submit:

    response = textgen_llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.write(response)
  