from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()


row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

t = '''
### Output indicators

Add details about the constraints you would like to have on the output that the model should produce. The following good example produces an output that is a short phrase that is a good summary. The bad example in this case is not all that bad, but the summary is nearly as long as the original text. Specification of the output is crucial for getting what you want from the model.
'''
row1_col1.title("Output indicators")

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

prompt = "Charles Mingus Jr. was an American jazz upright bassist, pianist, composer, bandleader, and author.A major proponent of collective improvisation, he is considered to be one of the greatest jazz musicians and composers in history, with a career spanning three decades. Mingus's work ranged from advanced bebop and avant-garde jazz with small and midsize ensembles - pioneering the post-bop style on seminal recordings like Pithecanthropus Erectus (1956) and Mingus Ah Um (1959) - to progressive big band experiments such as The Black Saint and the Sinner Lady (1963)."

prompt_type = st.selectbox(
    ":orange[Select Prompt Type]",("Prompt with clear output constraints indicator","Prompt without clear output specifications"))

if prompt_type == "Prompt with clear output constraints indicator":
    with st.form("myform1"):
        prompt_data = st.text_area(
        ":orange[User Prompt:]",
        height = 200,
        value = f"{prompt}\n\nPlease summarize the above text in one phrase."
        )
        submit = st.form_submit_button("Submit")
else:
    with st.form("myform2"):
        prompt_data = st.text_area(
        ":orange[User Prompt:]",
        height = 200,
        value = f"{prompt}\n\nPlease summarize the above text."
        )
        submit = st.form_submit_button("Submit")

if prompt_data and submit:

    response = textgen_llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.write(response)
  