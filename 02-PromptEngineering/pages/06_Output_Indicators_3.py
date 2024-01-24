from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Output indicators")

t = '''
### Output indicators

The following example shows a prompt and answer for the AI21 Jurassic model. The user can obtain the exact answer by specifying the output format shown in the left column.

'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Cohere','Meta','Amazon','Anthropic','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

# model_id=st.text_input('model_id',getmodelId(provider))

inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

textgen_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)

prompt = """Context: The NFL was formed in 1920 as the American Professional Football Association (APFA) before renaming itself the National Football League for the 1922 season. After initially determining champions through end-of-season standings, a playoff system was implemented in 1933 that culminated with the NFL Championship Game until 1966. Following an agreement to merge the NFL with the rival American Football League (AFL), the Super Bowl was first held in 1967 to determine a champion between the best teams from the two leagues and has remained as the final game of each NFL season since the merger was completed in 1970.\n\nQuestion: Based on the above context, when was the first Super Bowl?"""

prompt_type = st.selectbox(
    ":orange[Select Prompt Type]",("Prompt with clear output constraints indicator","Prompt without clear output specifications"))

if prompt_type == "Prompt with clear output constraints indicator":
    with st.form("myform1"):
        prompt_data = st.text_area(
        ":orange[User Prompt:]",
        height = 200,
        value = f"{prompt}\nPlease only output the year."
        )
        submit = st.form_submit_button("Submit")
else:
    with st.form("myform2"):
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
  