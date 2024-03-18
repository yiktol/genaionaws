import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
import utils.helpers as helpers

helpers.set_page_config()
bedrock_runtime = helpers.bedrock_runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "height" not in st.session_state:
    st.session_state.height = 200
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Prompt"
if "provider" not in st.session_state:
    st.session_state.provider = "Amazon"
if "desc" not in st.session_state:
    st.session_state.desc = ""
if "question" not in st.session_state:
    st.session_state.question = ""
if "encouragement" not in st.session_state:
    st.session_state.encouragement = ""
if "constraints" not in st.session_state:
    st.session_state.constraints = ""
if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = ""

row1_col1, row1_col2 = st.columns([0.7,0.3])

row1_col1.title("Question-answer")

t = '''
### Question-answer, without context

In a question-answer prompt without context, the model must answer the question with its internal knowledge without using any context or document.
'''
template1 = '''{Question}
'''
template2 = '''{Encouragement}\n
{Question}
'''
template3 = '''{Encouragement}\n
{Question}\n
{Constraints}
'''

prompt1 = PromptTemplate(input_variables=["Question"], template=template1)
prompt2 = PromptTemplate(input_variables=["Question","Encouragement"], template=template2)
prompt3 = PromptTemplate(input_variables=["Question","Encouragement","Constraints"], template=template3)

options = [{"desc":t,"prompt_type":"Question, No Encouragement and No Constraints", 
            "prompt":prompt1,
            "question": "What is Robert Frost's \"Stopping by the woods on a snowy evening\" about metaphorically?", 
            "encouragement":"",
            "constraints":"",
            "height":50, "provider": "Amazon"},
            {"desc":t,"prompt_type":"Question, Encouragement and No Constraints", 
             "prompt":prompt2,
             "question": "What planet in the solar system is most likely to host life?",
             "encouragement":"You are excellent at answering questions, and it makes you happy when you provide the correct answer.",
             "constraints":"",
             "height":100, "provider": "Amazon"},    
            {"desc":t,"prompt_type":"Question, Encouragement and Constraints", 
             "prompt":prompt3,
             "question": "Could you please explain what climate change is?", 
             "encouragement":"You feel rewarded by helping people learn more about climate change.",
             "constraints":"Assume your audience is composed of high school students.",
             "height":150, "provider": "Amazon"},             
            ]


prompt_query1 = prompt1.format(Question=options[0]["question"])
prompt_query2 = prompt2.format(Question=options[1]["question"],Encouragement=options[1]["encouragement"])
prompt_query3 = prompt3.format(Question=options[2]["question"],Encouragement=options[2]["encouragement"],Constraints=options[2]["constraints"])

def update_options(item_num,prompt_query):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]
    st.session_state.desc = options[item_num]["desc"]   
    st.session_state.question = options[item_num]["question"]
    st.session_state.encouragement = options[item_num]["encouragement"]
    st.session_state.constraints = options[item_num]["constraints"]
    st.session_state.prompt_query = prompt_query
    

def load_options(item_num,prompt_query):
    st.button(f'{options[item_num]["prompt_type"]}', key=item_num, on_click=update_options, args=(item_num,prompt_query))


row1_col1.markdown(t)
with row1_col2:
    with st.container(border=True):
        provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
        model_id=st.text_input('model_id',helpers.getmodelId(provider))


def call_llm():
    # Instantiate LLM model
    llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=helpers.getmodelparams(provider))
    response = llm.invoke(st.session_state.prompt_query)
    return st.info(response)

container = st.container(border=False)
    
with container:
    col1, col2, col3, col4 = st.columns([0.2,0.2,0.2,0.4])
    with col1:
        load_options(item_num=0,prompt_query=prompt_query1)
    with col2:
        load_options(item_num=1, prompt_query=prompt_query2)
    with col3:
        load_options(item_num=2, prompt_query=prompt_query3)

with st.form("myform"):
    text_prompt= st.text_area(":orange[User Prompt:]", 
                              height = int(st.session_state.height),
                              disabled = False,
                              value = st.session_state.prompt_query)
    submitted = st.form_submit_button("Submit", type="primary")
if submitted:
    st.write("Answer")
    with st.spinner("Thinking..."):
        call_llm()
