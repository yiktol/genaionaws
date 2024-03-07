from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)


row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Question or Instruction should be placed at the end")

t = '''
### The question or instruction should be placed at the end of the prompt for best results

Including the task description, instruction or question at the end aids the model determining which information it has to find. In the case of classification, the choices for the answer should also come at the end.

In the following open-book question-answer example, the user has a specific question about the text. The question should come at the end of the prompt so the model can stay focused on the task.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','Anthropic','AI21','Cohere','Meta'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()


llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)


prompt = '''Tensions increased after the 1911-1912 Italo-Turkish War demonstrated Ottoman weakness and led to the formation of the Balkan League, 
an alliance of Serbia, Bulgaria, Montenegro, and Greece.\n 
The League quickly overran most of the Ottomans' territory in the Balkans during the 1912-1913 First Balkan War, much to the surprise of outside observers.\n
The Serbian capture of ports on the Adriatic resulted in partial Austrian mobilization starting on 21 November 1912, including units along the Russian border in Galicia. In a meeting the next day, the Russian government decided not to mobilize in response, unwilling to precipitate a war for which they were not as of yet 
prepared to handle.\n\nWhich country captured ports?'''

with st.form("myform1"):
    prompt_data = st.text_area(
    ":orange[User Prompt:]",
    height = 290,
    value = prompt
    )
    submit = st.form_submit_button("Submit", type="primary")


if prompt_data and submit:

    response = llm(prompt_data)

    print(response)
    st.write("### Answer")
    st.info(response)
  