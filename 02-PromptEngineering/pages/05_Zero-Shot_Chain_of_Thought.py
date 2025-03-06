from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
import streamlit as st
import utils.helpers as helpers
import threading
# from streamlit.runtime.scriptrunner import add_script_run_ctx
# from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx


helpers.set_page_config()
bedrock_runtime = helpers.runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
    
st.sidebar.button(label='Clear Session Data', on_click=form_callback)
        
prompt1 = """Q: A juggler can juggle 16 balls. \
Half of the balls are golf balls, and half of the golf balls are blue. \
How many blue golf balls are there?\n 
A:
"""

prompt2 = """Q: A juggler can juggle 16 balls. \
Half of the balls are golf balls, and half of the golf balls are blue. \
How many blue golf balls are there?
(Think Step-by-Step)\n
A:"""


if "prompt" not in st.session_state:
    st.session_state.prompt = prompt1
if "height" not in st.session_state:
    st.session_state.height = 100
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Zero-shot"
if "image" not in st.session_state:
    st.session_state.image = "images/zero_shot.png"

row1_col1, row1_col2 = st.columns([0.7,0.3])

t = '''- Improves reasoning abilities in foundation models
- Addresses multi-step problem-solving challenges in arithmetic and commonsense reasoning task
- Generates intermediate reasoning steps, mimicking human train of thought, before providing the final answer.
- Enhances model performance on average compared to standard methods.
- Works better with larger models (>100B) and can be fine-tuned on CoT reasoning datasets for better interpretability.

'''
with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Chain-of-thought (CoT) Prompting")
        st.markdown(t)
    with col2:
        with st.container(border=True):
            provider = st.selectbox('Provider',helpers.list_providers, index=0)
            models = helpers.getmodelIds(provider)
            model_id = st.selectbox(
                'model', models, index=models.index(helpers.getmodelId(provider)))     

with st.expander("Zero Shot - Chain-of-thought (CoT)"):
    st.image("images/zero-shot-cot.png")

#Create the connection to Bedrock
def call_llm(prompt):
    llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=helpers.getmodelparams(provider))
    response = llm.invoke(prompt)
    # Print results
    return st.info(response)

def call_llm_chat(prompt):
    params = helpers.getmodelparams(provider)
    params.update({'messages':[{"role": "user", "content": prompt}]})
    
    llm = BedrockChat(model_id=model_id, client=bedrock_runtime, model_kwargs=params)
    response = llm.invoke(prompt)
    
    return st.info(response.content)

options = [{"prompt_type":"Zero-shot-CoT", "prompt": prompt2, "height":150,},
            {"prompt_type":"Zero-shot", "prompt": prompt1, "height":100}]


col1, col2 = st.columns(2)

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        prompt_data1 = st.text_area(f":orange[Zero-Shot:]", height = 150, value=prompt1 )
        submit = st.button("Submit", type="primary", key=1)

    with col2:
        prompt_data2 = st.text_area(f":orange[Zero-Shot-CoT:]", height = 150, value=prompt2 )
    


if submit:
    with st.spinner("Thinking..."):
        col1, col2 = st.columns(2)
        
        # ctx = get_script_run_ctx()
        with col1:
            if provider == "Claude 3":
                call_llm_chat(prompt_data1)
            else:
                # call_llm_chat(text_prompt)
                call_llm(prompt_data1)
        with col2:
            if provider == "Claude 3":
                call_llm_chat(prompt_data2)
            else:
                # call_llm_chat(text_prompt)
                call_llm(prompt_data2)
