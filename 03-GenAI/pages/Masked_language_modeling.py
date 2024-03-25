
import streamlit as st
from transformers import BertTokenizer, pipeline
from helpers import set_page_config

set_page_config()

st.title("BERT Masked language modeling (MLM)")

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
unmasker = pipeline('fill-mask', model='bert-base-uncased')

with st.form("myform"):
    prompt = st.text_input(
        "Enter prompt to tokenize",
        placeholder="Where is Himalayas in the world map?",
        value="Hello I'm a [MASK] model.",)
    submit = st.form_submit_button("Unmask", type='primary')


if prompt and submit:
    with st.spinner("Generating..."):
        unmask = unmasker(prompt)
        st.write("### Answer")
        st.write(unmask)


