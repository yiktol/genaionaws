
import streamlit as st
from transformers import BertTokenizer, pipeline
from helpers import set_page_config

set_page_config()

st.title("BERT Masked language modeling (MLM)")

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
unmasker = pipeline('fill-mask', model='bert-base-uncased')


dataset = [{"id": 0, "text": "A puppy is to dog as kitten is to [MASK]."},
           {"id": 1, "text": "The best part of this film is the [MASK] scene."},
           {"id": 2, "text": "I had a great time at the [MASK] restaurant."},
           {"id": 3, "text": "Hello I'm a [MASK] model."},
           {"id": 4, "text": "I would definitely visit this place again if I had the chance, it's [MASK]."},
           {"id": 5, "text": "The food at this restaurant was [MASK]."},
           {"id": 6, "text": "The atmosphere at the [MASK] place is so relaxing."},
           {"id": 7, "text": "I had a terrible experience at the [MASK] restaurant."},
           {"id": 8, "text": "The [MASK] movie was amazing!"}
           ]
    


def mlm_form(item_num):
    with st.form(f"myform-{item_num}"):
        prompt = st.text_input(
            "Enter prompt here",
            value=dataset[item_num]['text'],)
        submit = st.form_submit_button("Unmask", type='primary')

    return submit, prompt


def show(submit, prompt):

    if submit:
        with st.spinner("Generating..."):
            unmask = unmasker(prompt)
            st.write("### Answer")
            st.write(unmask)


tabs = st.tabs([f"Item {i+1}" for i in range(len(dataset))])

for i, tab in enumerate(tabs):
    with tab:
        submit, prompt = mlm_form(i)
        show(submit, prompt)

