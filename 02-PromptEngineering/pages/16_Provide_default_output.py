from langchain_community.llms import Bedrock
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Provide a default output")

t = '''
### Provide a default output that the model should present if it's unsure about the answers.

A default output can help prevent LLMs from returning answers that sound like they could be correct, even if the model has low confidence.
'''

row1_col1.markdown(t)
with row1_col2:
    with st.container(border=True):
        provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
        model_id=st.text_input('model_id',helpers.getmodelId(provider))


#Create the connection to Bedrock
bedrock_runtime = helpers.bedrock_runtime_client()

llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=helpers.getmodelparams(provider),
)

prompt = "Provide a proof of the Riemann hypothesis. If you don't know a proof, respond by saying \"I don't know.\""


with st.form("myform1"):
    prompt_data = st.text_area(
    ":orange[User Prompt:]",
    height = 50,
    value = f"{prompt}"
    )
    submit = st.form_submit_button("Submit",type="primary")


if prompt_data and submit:
    with st.spinner("Thinking..."):
        response = llm(prompt_data)

        # print(response)
        st.write("### Answer")
        st.info(response)
    