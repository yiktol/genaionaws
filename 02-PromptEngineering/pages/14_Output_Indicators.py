from langchain_community.llms import Bedrock
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock_runtime = helpers.bedrock_runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)


prompt1 = "Charles Mingus Jr. was an American jazz upright bassist, pianist, composer, bandleader, and author.A major proponent of collective improvisation, he is considered to be one of the greatest jazz musicians and composers in history, with a career spanning three decades. Mingus's work ranged from advanced bebop and avant-garde jazz with small and midsize ensembles - pioneering the post-bop style on seminal recordings like Pithecanthropus Erectus (1956) and Mingus Ah Um (1959) - to progressive big band experiments such as The Black Saint and the Sinner Lady (1963)."
prompt2 = "Human: Extract names and years: the term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence. The synonym self-teaching computers was also used in this time period."
prompt3 = "Context: The NFL was formed in 1920 as the American Professional Football Association (APFA) before renaming itself the National Football League for the 1922 season. After initially determining champions through end-of-season standings, a playoff system was implemented in 1933 that culminated with the NFL Championship Game until 1966. Following an agreement to merge the NFL with the rival American Football League (AFL), the Super Bowl was first held in 1967 to determine a champion between the best teams from the two leagues and has remained as the final game of each NFL season since the merger was completed in 1970.\n\nQuestion: Based on the above context, when was the first Super Bowl?"


row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

t1 = '''Add details about the constraints you would like to have on the output that the model should produce. \
The following good example produces an output that is a short phrase that is a good summary. \
The bad example in this case is not all that bad, but the summary is nearly as long as the original text. \
Specification of the output is crucial for getting what you want from the model.
'''
t2 = '''Here we give some additional examples from Claude and AI21 Jurassic models using output indicators. \
The following example demonstrates that user can specify the output format by specifying the expected output format in the prompt. \
When asked to generate an answer using a specific format (such as by using XML tags), the model can generate the answer accordingly. \
Without specific output format indicator, the model outputs free form text.'''
t3 = '''The following example shows a prompt and answer for the AI21 Jurassic model. \
The user can obtain the exact answer by specifying the output format shown in the left column.
'''

options = [{"prompt_type":"With Output Indicator", "prompt": "Please summarize the above text in one phrase.", "height":210, "provider": "Amazon"},
            {"prompt_type":"No Output Indicator", "prompt": "Please summarize the above text.", "height":210, "provider": "Amazon"},
            {"prompt_type":"With Output Indicator", "prompt": "Please generate answer in <name></name> and <year></year> tags.", "height":210, "provider": "Anthropic"},
            {"prompt_type":"No Output Indicator", "prompt": "", "height":210,"provider": "Anthropic"},
            {"prompt_type":"With Output Indicator", "prompt": "Please only output the year.", "height":280,"provider": "AI21"},
            {"prompt_type":"No Output Indicator", "prompt": "", "height":250,"provider": "AI21"},
            
            ]

if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "height" not in st.session_state:
    st.session_state.height = 210
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Prompt"
if "provider" not in st.session_state:
    st.session_state.provider = "Amazon"

row1_col1.title("Output indicators")

with row1_col2:
    with st.container(border=True):
        provider = st.text_input('Provider',st.session_state.provider)
        model_id=st.text_input('model_id',helpers.getmodelId(provider))


def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]

def load_options(item_num):
    st.button(f'{options[item_num]["prompt_type"]}', key=item_num, on_click=update_options, args=(item_num,))

llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=helpers.getmodelparams(provider))
    
tab1, tab2, tab3 = row1_col1.tabs(["Example1", "Example2", "Example3"])
with tab1:

    st.markdown(t1)
    container = st.container(border=False)
        
    with container:
        col1, col2, col3 = st.columns([0.2,0.2,0.6])
        with col1:
            load_options(item_num=0)
        with col2:
            load_options(item_num=1)

    with st.form("form1"):
        prompt_data = st.text_area(f":orange[{st.session_state.prompt_type}:]", height = int(st.session_state.height), value = f"{prompt1}\n\n{st.session_state.prompt}")
        submit = st.form_submit_button("Submit", type="primary")

    if submit:
        with st.spinner("Thinking..."):
            response = llm(prompt_data)

            #print(response)
            st.write("### Answer")
            st.info(response)

with tab2:
    st.markdown(t2)
    container = st.container(border=False)
        
    with container:
        col1, col2, col3 = st.columns([0.2,0.2,0.6])
        with col1:
            load_options(item_num=2)
        with col2:
            load_options(item_num=3)

    with st.form("form2"):
        prompt_data2 = st.text_area(f":orange[{st.session_state.prompt_type}:]", height = int(st.session_state.height), value = f"{prompt2}\n\n{st.session_state.prompt}\n\nAssistant:")
        submit2 = st.form_submit_button("Submit", type="primary")

    if submit2:
        with st.spinner("Thinking..."):
            response = llm(prompt_data2)

            #print(response)
            st.write("### Answer")
            st.info(response)

with tab3:
    st.markdown(t3)
    container = st.container(border=False)
        
    with container:
        col1, col2, col3 = st.columns([0.2,0.2,0.6])
        with col1:
            load_options(item_num=4)
        with col2:
            load_options(item_num=5)

    with st.form("form3"):
        prompt_data3 = st.text_area(f":orange[{st.session_state.prompt_type}:]", height = int(st.session_state.height), value = f"{prompt3}\n\n{st.session_state.prompt}")
        submit3 = st.form_submit_button("Submit", type="primary")

    if submit3:
        with st.spinner("Thinking..."):
            response = llm(prompt_data3)

            #print(response)
            st.write("### Answer")
            st.info(response)   