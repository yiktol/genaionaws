import streamlit as st
from jinja2 import Environment, FileSystemLoader
import utils.helpers


def set_page_config():
    st.set_page_config( 
    page_title="Bedrock Foundation Models",  
    page_icon=":rock:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initsessionkeys(dataset,suffix):
    for key in dataset.keys():
        if key not in st.session_state[suffix]:
            st.session_state[suffix][key] = dataset[key]
    return st.session_state[suffix]

def update_options(dataset,suffix,item_num):
    for key in dataset[item_num]:
        st.session_state[suffix][key] = dataset[item_num][key]

def load_options(dataset,suffix, item_num):    
    st.write("Prompt:",dataset[item_num]["prompt"])
    if "negative_prompt" in dataset[item_num].keys():
        st.write("Negative Prompt:", dataset[item_num]["negative_prompt"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(dataset,suffix,item_num))  
    

def reset_session():
    def form_callback():
        for key in st.session_state.keys():
            del st.session_state[key]


    st.button(label='Reset', on_click=form_callback)




def render_claude_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], 
        max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], 
        top_p=st.session_state['top_p'],
        top_k = st.session_state['top_k'],
        model = st.session_state['model'])
    return output

def render_cohere_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], 
        max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], 
        top_p=st.session_state['top_p'],
        top_k = st.session_state['top_k'],
        model = st.session_state['model'])
    return output

def render_meta_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], top_p=st.session_state['top_p'],
        model = st.session_state['model'])
    return output

def render_mistral_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], max_tokens=st.session_state['max_tokens'], 
        temperature=st.session_state['temperature'], top_p=st.session_state['top_p'],
        model = st.session_state['model'])
    return output

def render_stabilityai_code(templatePath):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state['prompt'], cfg_scale=st.session_state['cfg_scale'], 
        seed=st.session_state['seed'], steps=st.session_state['steps'],
        model = st.session_state['model'])
    return output


def create_two_tabs(dataset,suffix):
    tab1, tab2 = st.tabs(["Prompt1", "Prompt2"])
    with tab1:
        load_options(dataset,suffix,item_num=0)
    with tab2:
        load_options(dataset,suffix,item_num=1)
 
def create_three_tabs(dataset,suffix):
    tab1, tab2, tab3 = st.tabs(["Prompt1", "Prompt2", "Prompt3"])
    with tab1:
        load_options(dataset,suffix,item_num=0)
    with tab2:
        load_options(dataset,suffix,item_num=1) 
    with tab3:
        load_options(dataset,suffix, item_num=2)

def create_four_tabs(dataset,suffix):
    tab1, tab2, tab3, tab4 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4"])
    with tab1:
        load_options(dataset,suffix, item_num=0)
    with tab2:
        load_options(dataset,suffix, item_num=1)
    with tab3:
        load_options(dataset,suffix, item_num=2)
    with tab4:
        load_options(dataset,suffix, item_num=3)

def create_five_tabs(dataset,suffix):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"])
    with tab1:
        load_options(dataset, suffix, item_num=0)
    with tab2:
        load_options(dataset, suffix, item_num=1)
    with tab3:
        load_options(dataset, suffix, item_num=2)
    with tab4:
        load_options(dataset, suffix, item_num=3)
    with tab5:
        load_options(dataset, suffix, item_num=4)

def create_six_tabs(dataset,suffix):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6"])
    with tab1:
        load_options(dataset,suffix, item_num=0)
    with tab2:
        load_options(dataset,suffix, item_num=1)
    with tab3:
        load_options(dataset,suffix, item_num=2)
    with tab4:
        load_options(dataset,suffix, item_num=3)
    with tab5:
        load_options(dataset,suffix, item_num=4)
    with tab6:
        load_options(dataset,suffix, item_num=5)

def create_seven_tabs(dataset,suffix):
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6", "Prompt7"])
    with tab1:
        load_options(dataset,suffix, item_num=0)
    with tab2:
        load_options(dataset,suffix, item_num=1)
    with tab3:
        load_options(dataset,suffix, item_num=2)
    with tab4:
        load_options(dataset,suffix, item_num=3)
    with tab5:
        load_options(dataset,suffix, item_num=4)
    with tab6:
        load_options(dataset,suffix, item_num=5)
    with tab7:
        load_options(dataset,suffix, item_num=6)

def create_tabs(dataset,suffix):
    if len(dataset) == 2:
        create_two_tabs(dataset,suffix)
    elif len(dataset) == 3:
        create_three_tabs(dataset,suffix)
    elif len(dataset) == 4:
        create_four_tabs(dataset,suffix)
    elif len(dataset) == 5:
        create_five_tabs(dataset,suffix)
    elif len(dataset) == 6:
        create_six_tabs(dataset,suffix)
    elif len(dataset) == 7:
        create_seven_tabs(dataset,suffix)
        
