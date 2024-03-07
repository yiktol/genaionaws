from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
bedrock_runtime = bedrock_runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
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


if "prompt" not in st.session_state:
    st.session_state.prompt = prompt1
if "height" not in st.session_state:
    st.session_state.height = 100
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Zero-shot"

row1_col1, row1_col2 = st.columns([0.7,0.3])

row1_col1.title("Few-shot vs. zero-shot")

t = '''
### Few-shot prompting vs. zero-shot prompting

It is sometimes useful to provide a few examples to help LLMs better calibrate their output to meet your expectations, also known as few-shot prompting or in-context learning, where a shot corresponds to a paired example input and the desired output. To illustrate, first here is an example of a zero-shot sentiment classification prompt where no example input-output pair is provided in the prompt text:
'''
row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','Anthropic','Cohere','Meta'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button('Set Parameters') 

row1_col2.button(label='Clear Session Data', key="clear",on_click=form_callback)

#Create the connection to Bedrock
llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider),
)

options = [{"prompt_type":"Few-shot", "prompt": prompt2, "height":350},
            {"prompt_type":"Zero-shot", "prompt": prompt1, "height":100}]

def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]

def load_options(item_num):
    st.button(f'{options[item_num]["prompt_type"]}', on_click=update_options, args=(item_num,))



container = st.container(border=False)
    
with container:
    col1, col2, col3 = st.columns([0.1,0.1,0.8])
    with col1:
        load_options(item_num=0)
    with col2:
        load_options(item_num=1)

    


with st.form("myform1"):
    prompt_data = st.text_area(f":orange[{st.session_state.prompt_type}:]", height = int(st.session_state.height), key="prompt" )
    submit = st.form_submit_button("Submit", type="primary")


if submit:
    with st.spinner("Thinking..."):
        response = llm(prompt_data)
        #print(response)
        st.write("### Answer")
        st.info(response)
        