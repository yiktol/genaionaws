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
        
prompt1 = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. \
How many tennis balls does he have now?\n
A: The answer is 11.\n
Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?\n
A:
"""

prompt2 = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. \
Each can has 3 tennis balls. How many tennis balls does he have now?\n
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.\n
Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?\n
A:"""


if "prompt" not in st.session_state:
    st.session_state.prompt = prompt1
if "height" not in st.session_state:
    st.session_state.height = 150
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Zero-shot"
if "image" not in st.session_state:
    st.session_state.image = "images/zero_shot.png"

row1_col1, row1_col2 = st.columns([0.7,0.3])

t = '''Works best with larger models
Effective with:
- Arithmetic
- Common Sense
- Symbolic reasoning
'''
with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Few-Shot Chain-of-thought (CoT)")
        st.markdown(t)
    with col2:
        with st.container(border=True):
            provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
            model_id=st.text_input('model_id',helpers.getmodelId(provider))        

with st.expander("Few-Shot- Chain-of-thought (CoT)"):
    st.image("images/few_shot_cot.png")

#Create the connection to Bedrock
llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=helpers.getmodelparams(provider),
)

options = [{"prompt_type":"Few-shot-CoT", "prompt": prompt2, "height":250,},
            {"prompt_type":"Few-shot", "prompt": prompt1, "height":250}]

    
def call_llm(prompt_data):
    response = llm.invoke(prompt_data)
    st.write("### Answer")
    st.info(response)

col1, col2 = st.columns(2)

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        prompt_data1 = st.text_area(f":orange[Few-Shot:]", height = 250, value=prompt1 )
        submit = st.button("Submit", type="primary", key=1)

    with col2:
        prompt_data2 = st.text_area(f":orange[Few-Shot-CoT:]", height = 250, value=prompt2 )
    


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

        