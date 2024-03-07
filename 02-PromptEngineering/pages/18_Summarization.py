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

row1_col1.title("Summarization")

t1 = '''
For a summarization task, the prompt is a passage of text, and the model must respond with a shorter passage that captures the main points of the input. Specification of the output in terms of length (number of sentences or paragraphs) is helpful for this use case.
'''
t2 = '''
In the following example, Claude summarizes the given text in one sentence. To include input text in your prompts, format the text with XML mark up: <text> {text content} </text>. Using XML within prompts is a common practice when prompting Claude models.
'''
template1 = '''The following is text from a {Category}:\n
{Context}\n
Summarize the above {Category} in {length_of_summary}
'''
template2 = '''Human: Please read the text:\n
<text>
{Context}
</text>\n
Summarize the text in {length_of_summary}\n
Assistant:
'''

options = [{"desc":t1,"prompt_type":"Summarization 1", 
            "category": "restaurant review", 
            "context":'''I finally got to check out Alessandro\'s Brilliant Pizza \
and it is now one of my favorite restaurants in Seattle. \
The dining room has a beautiful view over the Puget Sound \
but it was surprisingly not crowed. I ordered the fried \
castelvetrano olives, a spicy Neapolitan-style pizza \
and a gnocchi dish. The olives were absolutely decadent, \
and the pizza came with a smoked mozzarella, which was delicious. \
The gnocchi was fresh and wonderful. The waitstaff were attentive, \
and overall the experience was lovely. I hope to return soon.''',
            "length_of_summary":"one sentence",
            "height":230, "provider": "Amazon"},
            {"desc":t2,"prompt_type":"Summarization 2", 
            "context":'''In game theory, the Nash equilibrium, named after the mathematician John Nash, is the most common way to define the solution of a non-cooperative game involving two or more players. \
In a Nash equilibrium, each player is assumed to know the equilibrium strategies of the other players, and no one has anything to gain by changing only one's own strategy. \
The principle of Nash equilibrium dates back to the time of Cournot, who in 1838 applied it to competing firms choosing outputs.''',
            "length_of_summary":"one sentence",
            "height":300, "provider": "Anthropic"},]



if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "height" not in st.session_state:
    st.session_state.height = 200
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = options[0]["prompt_type"]
if "provider" not in st.session_state:
    st.session_state.provider = "Amazon"
if "desc" not in st.session_state:
    st.session_state.desc = options[0]["desc"] 
if "category" not in st.session_state:
    st.session_state.category = options[0]["category"]
if "context" not in st.session_state:
    st.session_state.context = options[0]["context"]
if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = ""


def update_options(item_num,prompt_query):
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]
    st.session_state.desc = options[item_num]["desc"]   
    st.session_state.context = options[item_num]["context"]
    st.session_state.prompt_query = prompt_query
    

def load_options(item_num,prompt_query):
    st.button(f'{options[item_num]["prompt_type"]}', key=item_num, on_click=update_options, args=(item_num,prompt_query))

prompt1 = PromptTemplate(input_variables=["Category","Context","length_of_summary"], template=template1)
prompt_query1 = prompt1.format(
        Category=options[0]["category"],
        Context=options[0]["context"],
        length_of_summary=options[0]["length_of_summary"])

prompt2 = PromptTemplate(input_variables=["Context","length_of_summary"], template=template2)
prompt_query2 = prompt2.format(
        Context=options[1]["context"],
        length_of_summary=options[1]["length_of_summary"]
        )


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
    col1, col2, col3= st.columns([0.2,0.2,0.6])
    with col1:
        load_options(item_num=0,prompt_query=prompt_query1)
    with col2:
        load_options(item_num=1,prompt_query=prompt_query2)

with st.form("myform"):
    text_prompt = st.text_area(":orange[User Prompt:]", 
                              height = int(st.session_state.height),
                              disabled = False,
                              value = st.session_state.prompt_query)
    submitted = st.form_submit_button("Submit")
if text_prompt and submitted:
    st.write("Answer")
    with st.spinner("Thinking..."):
        call_llm()

