
import streamlit as st
from transformers import BertTokenizer
from helpers import set_page_config

set_page_config()

st.title("BERT Tokenizer")

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

with st.form("myform"):
    prompt = st.text_input(
        "Enter prompt to tokenize",
        placeholder="Where is Himalayas in the world map?",
        value="Where is Himalayas in the world map?",)
    submit = st.form_submit_button("Tokenize", type='primary')


if prompt and submit:
    encoding = tokenizer.encode(prompt)
    # print(encoding)
    # print(tokenizer.convert_ids_to_tokens(encoding))
    # output 1: [101, 2073, 2003, 26779, 1999, 1996, 2088, 4949, 1029, 102]
    # output 2: ['[CLS]', 'where', 'is', 'himalayas', 'in', 'the', 'world', 'map', '?', '[SEP]']

    d = {}
    
    st.write("### Answer")
    # print ("{:<25} {:<15}".format('Word','Token'))
    for i in range(0,len(encoding)):
        # print("{:<25} {:<15}".format(tokenizer.decode(encoding[i]),encoding[i]))
        d[tokenizer.decode(encoding[i])] = encoding[i]
    st.table(d)

