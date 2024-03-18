from langchain_community.llms import Bedrock
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock_runtime = helpers.bedrock_runtime_client()

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

t = '''LLM models can provide clear steps for certain tasks, and including the phrase \
Think step-by-step to come up with the right answer can help produce the appropriate output.
'''

options = [{"prompt_type":"Example1", "prompt": prompt1, "height":120, "provider": "AI21"},
            {"prompt_type":"Example2", "prompt": prompt2, "height":210, "provider": "Anthropic"},            
            ]

def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]

def load_options(item_num):
    st.write(f'Prompt: {options[item_num]["prompt"]}')
    st.button(f'Load Prompt', key=item_num, on_click=update_options, args=(item_num,))

with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Answer step by step")
        st.subheader("Complex tasks: build toward the answer step by step")
        st.markdown(t)
        with st.form("myform2"):
            prompt_data = st.text_area(f":orange[{st.session_state.prompt_type}:]", height = int(st.session_state.height),key="prompt" )
            submit = st.form_submit_button("Submit", type="primary")
    with col2:
        with st.container(border=True):
            provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
            model_id=st.text_input('model_id',helpers.getmodelId(provider))
        with st.container(border=True):
            tab1, tab2 = st.tabs(["Example1", "Example2"])
            with tab1:
                load_options(item_num=0)
            with tab2:
                load_options(item_num=1)



llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=helpers.getmodelparams(provider))

if prompt_data and submit:
    with st.spinner("Thinking..."):
        response = llm.invoke(prompt_data)

        #print(response)
        st.write("### Answer")
        st.info(response)
    