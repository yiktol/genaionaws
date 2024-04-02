from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock_runtime = helpers.runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
    
st.sidebar.button(label='Clear Session Data', on_click=form_callback)
        
prompt1 = """Tell me the sentiment of the following headline and categorize it as either positive, negative or neutral:\n
\"New airline between Seattle and San Francisco offers a great opportunity for both passengers and investors.\""""

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
if "image" not in st.session_state:
    st.session_state.image = "images/zero_shot.png"

row1_col1, row1_col2 = st.columns([0.7,0.3])

t = '''
### Few-shot prompting vs. zero-shot prompting

It is sometimes useful to provide a few examples to help LLMs better calibrate their output to meet your expectations, also known as few-shot prompting or in-context learning, where a shot corresponds to a paired example input and the desired output. To illustrate, first here is an example of a zero-shot sentiment classification prompt where no example input-output pair is provided in the prompt text:
'''
 
#Create the connection to Bedrock


options = [{"prompt_type":"Few-shot", "prompt": prompt2, "height":350,"image":"images/few_shot.png"},
            {"prompt_type":"Zero-shot", "prompt": prompt1, "height":150,"image":"images/zero_shot.png"}]

def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.image = options[item_num]["image"]

def load_options(item_num):
    st.write(f'Prompt: {options[item_num]["prompt"]}')
    st.button(f'Load Prompt', key=item_num, on_click=update_options, args=(item_num,))



with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Few-shot vs. Zero-shot")
        st.markdown(t)
        with st.expander(st.session_state.prompt_type):
            st.image(st.session_state.image)
        with st.form("myform2"):
            prompt_data = st.text_area(f":orange[{st.session_state.prompt_type}:]", height = int(st.session_state.height),key="prompt" )
            submit = st.form_submit_button("Submit", type="primary")
    with col2:
        with st.container(border=True):
            provider = st.selectbox('Provider',helpers.list_providers, index=2)
            models = helpers.getmodelIds(provider)
            model_id = st.selectbox(
                'model', models, index=models.index(helpers.getmodelId(provider)))  
        with st.container(border=True):
            tab1, tab2 = st.tabs(["Few-shot", "Zero-shot"])
            with tab1:
                load_options(item_num=0)
            with tab2:
                load_options(item_num=1)

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

if submit:
    with st.spinner("Thinking..."):
        if provider == "Claude 3":
            call_llm_chat(prompt_data)
        else:
            # call_llm_chat(text_prompt)
            call_llm(prompt_data)