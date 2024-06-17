
import streamlit as st
from transformers import BertTokenizer
from helpers import set_page_config

set_page_config()

st.title("BERT Tokenizer")

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


dataset = [{"id": 0, "text": "Where is Himalayas in the world map?"},
           {"id": 1, "text": "What is the capital of France?"},
           {"id": 2, "text": "Who is the president of the United States?"},
           {"id": 3, "text": "What is the largest planet in our solar system?"},
           {"id": 4, "text": "What is the smallest country in the world?"},       
           
           
           ]


def token_form(item_num):
    with st.form(f"myform-{item_num}"):
        prompt = st.text_input(
            "Enter prompt to tokenize",
            value=dataset[item_num]['text'],)
        submit = st.form_submit_button("Tokenize", type='primary')
        
    return submit, prompt


def show(submit, prompt):

    if prompt and submit:
        encoding = tokenizer.encode(prompt)

        d = {}
        
        st.write("### Answer")
        # print ("{:<25} {:<15}".format('Word','Token'))
        for i in range(0,len(encoding)):
            # print("{:<25} {:<15}".format(tokenizer.decode(encoding[i]),encoding[i]))
            d[tokenizer.decode(encoding[i])] = encoding[i]
        st.table(d)
        

tabs = st.tabs([f"Item {i+1}" for i in range(len(dataset))])

for i, tab in enumerate(tabs):
    with tab:
        submit, prompt = token_form(i)
        show(submit, prompt)
