import json
import streamlit as st
import utils.helpers as helpers


helpers.set_page_config()

st.title("Prepare Datsets")
st.markdown("""### Dataset Card for CNN Dailymail Dataset
#### Dataset Summary

The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. \
The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering.  
            """)

st.code("""{
"id": "0054d6d30dbcad772e20b22771153a2a9cbeaf62",
"article": "(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. \
The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. \
The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour."
"highlights": "The elderly woman suffered from diabetes and hypertension, ship's doctors say. Previously, 86 passengers had fallen ill on the ship, Agencia Brasil says."
}
""", language='json')

with st.expander(":orange[Prepare the training dataset.]"):
    st.markdown("""a. Create a training dataset file containing the following one line and name it train.jsonl.""")

    st.code("""{"prompt": "what is AWS", "completion": "it's Amazon Web Services"}""", language='json')

    st.markdown("""b. Create an S3 bucket for your training data and another one for your output data (the names must be unique).""")
    st.markdown("""c. Upload train.jsonl into the training data bucket""")


st.subheader("Train Dataset")

train = helpers.load_jsonl("data/train-cnn-5K.jsonl")

st.code(json.dumps(train[0:3], indent=4), language='json')

st.subheader("Validation Dataset")

train = helpers.load_jsonl("data/validation-cnn-1K.jsonl")

st.code(json.dumps(train[0:3], indent=4), language='json')

st.subheader("Test Dataset")

train = helpers.load_jsonl("data/test-cnn-10.jsonl")

st.code(json.dumps(train[0:3], indent=4), language='json')