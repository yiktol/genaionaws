import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
bedrock_runtime = bedrock_runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

row1_col1, row1_col2 = st.columns([0.7,0.3])

row1_col1.title("Text generation")

t1 = '''
Given a prompt, LLMs on Amazon Bedrock can respond with a passage of original text that matches the description. Here is one example:
'''
t2 = '''
For text generation use cases, specifying detailed task requirements can work well. In the following example, we ask the model to generate response with exclamation points.
'''
t3 = '''
In the following example, a user prompts the model to take on the role of a specific person when generating the text. Notice how the signature reflects the role the model is taking on in the response.
'''
template1 = '''Please write a {Text_Category} in the voice of {Role}.'''
template2 = '''{Task_specification}\nPlease write a {Text_Category} in the voice of {Role}.'''
template3 = '''{Role_assumption}\n{Task_description}.'''


options = [{"desc":t1,"prompt_type":"Text generation 1", 
            "category": "email", 
            "role":"friend",
            "task":"",
            "height":50, "provider": "Amazon"},
            {"desc":t2,"prompt_type":"Text generation 2", 
            "category": "email", 
            "role":"friend",
            "task":"Write text with exclamation points.",
            "height":50, "provider": "Amazon"},
            {"desc":t3,"prompt_type":"Text generation 3", 
            "category": "", 
            "role":"My name is Jack.",
            "task":"Help me write a note expressing my gratitude to my parents for taking my son (their grandson) to the zoo. I miss my parents so much.",
            "height":50, "provider": "Amazon"},]


prompt1 = PromptTemplate(input_variables=["Text_Category","Role"], template=template1)
prompt_query1 = prompt1.format(Text_Category=options[0]["category"],Role=options[0]["role"])
prompt2 = PromptTemplate(input_variables=["Text_Category","Role","Task_specification"], template=template2)
prompt_query2 = prompt2.format(Text_Category=options[1]["category"],Task_specification=options[1]["task"],Role=options[1]["role"])
prompt3 = PromptTemplate(input_variables=["Role_assumption","Task_description"], template=template3)
prompt_query3 = prompt3.format(Role_assumption=options[2]["role"],Task_description=options[2]["task"])


if "height" not in st.session_state:
    st.session_state.height = 50
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = options[0]["prompt_type"]
if "provider" not in st.session_state:
    st.session_state.provider = "Amazon"
if "desc" not in st.session_state:
    st.session_state.desc = options[0]["desc"] 
if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = prompt_query1

def update_options(item_num,prompt_query):
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]
    st.session_state.desc = options[item_num]["desc"]   
    st.session_state.prompt_query = prompt_query
    
def load_options(item_num,prompt_query):
    st.button(f'{options[item_num]["prompt_type"]}', key=item_num, on_click=update_options, args=(item_num,prompt_query))

row1_col1.markdown(st.session_state.desc)
with row1_col2:
    with st.container(border=True):
        provider = st.text_input('Provider',st.session_state.provider )
        model_id=st.text_input('model_id',getmodelId(st.session_state.provider))
    
def call_llm():
    llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=getmodelparams(provider))
    response = llm.invoke(st.session_state.prompt_query)
    # Print results
    return st.info(response)

container = st.container(border=False)
    
with container:
    col1, col2, col3,col4= st.columns([0.2,0.2,0.2,0.4])
    with col1:
        load_options(item_num=0,prompt_query=prompt_query1)
    with col2:
        load_options(item_num=1,prompt_query=prompt_query2)
    with col3:
        load_options(item_num=2,prompt_query=prompt_query3)

with st.form("myform"):
    text_prompt = st.text_area(":orange[User Prompt:]", height = int(st.session_state.height), value = st.session_state.prompt_query)
    submitted = st.form_submit_button("Submit", type="primary")
    
if submitted:
    st.write("Answer")
    with st.spinner("Thinking..."):
        call_llm()
