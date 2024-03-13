import streamlit as st
import jsonlines


def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d

def initsessionkeys(dataset):
    for key in dataset.keys():
        # print(key)
        if key not in st.session_state:
            st.session_state[key] = dataset[key]
    # print(st.session_state)
    return st.session_state

def update_options(dataset,item_num):
    for key in dataset[item_num]:
        if key in ["model","temperature","top_p","top_k","max_tokens"]:
            continue
        else:
            st.session_state[key] = dataset[item_num][key]
        # print(key, dataset[item_num][key])

def load_options(dataset,item_num):    
    # dataset = load_jsonl('mistral.jsonl')
    st.write("Prompt:",dataset[item_num]["prompt"])
    if "negative_prompt" in dataset[item_num].keys():
        st.write("Negative Prompt:", dataset[item_num]["negative_prompt"])
    st.button("Load Prompt", key=item_num, on_click=update_options, args=(dataset,item_num))  


def create_two_tabs(dataset):
    tab1, tab2 = st.tabs(["Prompt1", "Prompt2"])
    with tab1:
        load_options(dataset,item_num=0)
    with tab2:
        load_options(dataset,item_num=1)
 
def create_three_tabs(dataset):
    tab1, tab2, tab3 = st.tabs(["Prompt1", "Prompt2", "Prompt3"])
    with tab1:
        load_options(dataset,item_num=0)
    with tab2:
        load_options(dataset,item_num=1) 
    with tab3:
        load_options(dataset, item_num=2)

def create_four_tabs(dataset):
    tab1, tab2, tab3, tab4 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)

def create_five_tabs(dataset):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)
    with tab5:
        load_options(dataset, item_num=4)

def create_six_tabs(dataset):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)
    with tab5:
        load_options(dataset, item_num=4)
    with tab6:
        load_options(dataset, item_num=5)

def create_seven_tabs(dataset):
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4", "Prompt5", "Prompt6", "Prompt7"])
    with tab1:
        load_options(dataset, item_num=0)
    with tab2:
        load_options(dataset, item_num=1)
    with tab3:
        load_options(dataset, item_num=2)
    with tab4:
        load_options(dataset, item_num=3)
    with tab5:
        load_options(dataset, item_num=4)
    with tab6:
        load_options(dataset, item_num=5)
    with tab7:
        load_options(dataset, item_num=6)

def create_tabs(dataset): 
    st.subheader('Prompt Examples:')
    container2 = st.container(border=True)
    with container2:    
        if len(dataset) == 2:
            create_two_tabs(dataset)
        elif len(dataset) == 3:
            create_three_tabs(dataset)
        elif len(dataset) == 4:
            create_four_tabs(dataset)
        elif len(dataset) == 5:
            create_five_tabs(dataset)
        elif len(dataset) == 6:
            create_six_tabs(dataset)
        elif len(dataset) == 7:
            create_seven_tabs(dataset)