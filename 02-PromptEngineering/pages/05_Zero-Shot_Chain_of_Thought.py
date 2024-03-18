from langchain_community.llms import Bedrock
import streamlit as st
import utils.helpers as helpers
import threading

helpers.set_page_config()
bedrock_runtime = helpers.bedrock_runtime_client()

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
            provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
            model_id=st.text_input('model_id',helpers.getmodelId(provider))        

with st.expander("Zero Shot - Chain-of-thought (CoT)"):
    st.image("images/zero-shot-cot.png")

#Create the connection to Bedrock
llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=helpers.getmodelparams(provider),
)

options = [{"prompt_type":"Zero-shot-CoT", "prompt": prompt2, "height":150,},
            {"prompt_type":"Zero-shot", "prompt": prompt1, "height":100}]

    
def call_llm(prompt_data):
    response = llm.invoke(prompt_data)
    st.write("### Answer")
    st.info(response)

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
        
        with col1:
            t1 = threading.Thread(target=call_llm(prompt_data1))
        with col2:
            t2 = threading.Thread(target=call_llm(prompt_data2))


        t1.start()
        t2.start()
        
        t2.join()
        t1.join()
        #print(response)

        