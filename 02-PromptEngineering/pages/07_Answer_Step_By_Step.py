from langchain_community.llms import Bedrock
import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
bedrock_runtime = bedrock_runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

prompt1 = "At a Halloween party, Jack gets 15 candies. Jack eats 5 candies. He wants to give each friend 5 candies. How many friends can receive candies?\n\nThink step-by-step to come up with the right answer."
prompt2 = "Human: A right triangle has a side of length 5 and a hypotenuse of length 13. What is the length of the other side?\n\nAssistant: Can I think step-by-step?\n\nHuman: Yes, please do.\n\nAssistant:"


if "prompt" not in st.session_state:
    st.session_state.prompt = prompt1
if "height" not in st.session_state:
    st.session_state.height = 120
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Prompt"
if "provider" not in st.session_state:
    st.session_state.provider = "AI21"


row1_col1, row1_col2 = st.columns([0.7,0.3])


row1_col1.title("Answer step by step")
row1_col1.subheader("Complex tasks: build toward the answer step by step")

t = '''LLM models can provide clear steps for certain tasks, and including the phrase \
Think step-by-step to come up with the right answer can help produce the appropriate output.
'''
row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.text_input('Provider',st.session_state.provider)
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 


options = [{"prompt_type":"Example1", "prompt": prompt1, "height":120, "provider": "AI21"},
            {"prompt_type":"Example2", "prompt": prompt2, "height":210, "provider": "Anthropic"},            
            ]

def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]

def load_options(item_num):
    st.button(f'{options[item_num]["prompt_type"]}', key=item_num, on_click=update_options, args=(item_num,))

llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=getmodelparams(st.session_state.provider))

container = st.container(border=False)
    
with container:
    col1, col2, col3 = st.columns([0.1,0.1,0.8])
    with col1:
        load_options(item_num=0)
    with col2:
        load_options(item_num=1)

with st.form("myform1"):
    prompt_data = st.text_area(":orange[User Prompt:]", height = int(st.session_state.height), key="prompt" )
    submit = st.form_submit_button("Submit", type="primary")


if prompt_data and submit:
    with st.spinner("Thinking..."):
        response = llm.invoke(prompt_data)

        #print(response)
        st.write("### Answer")
        st.info(response)
    