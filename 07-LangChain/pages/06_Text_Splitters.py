import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Bedrock
from helpers import bedrock_runtime_client, set_page_config


set_page_config()


st.header("Text Splitters")
st.markdown("""Once you've loaded documents, you'll often want to transform them to better suit your application. \
The simplest example is you may want to split a long document into smaller chunks that can fit into your model's context window. \
LangChain has a number of built-in document transformers that make it easy to split, combine, filter, and otherwise manipulate documents.
            """)

st.subheader(":orange[Recursively split by character]")
st.markdown("""This text splitter is the recommended one for generic text. \
It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. \
The default list is ["\\n\\n", "\\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, \
and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text. \
[Sample TXT file](https://training.yikyakyuk.com/genai/docs/alice.txt)""")

expander = st.expander("See code")
expander.code("""from langchain.text_splitter import RecursiveCharacterTextSplitter
# This is a long document we can split up.
with open("alice.txt") as f:
    alice_in_wonderland = f.read()
    
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)   

texts = text_splitter.create_documents([alice_in_wonderland])
print(texts[0])
print(texts[1]) 
    """,language="python")


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

with st.form("form1"):
    uploaded_file = st.file_uploader("Choose a TXT file:", type=['txt'])
    submit = st.form_submit_button("Split TXT",type="primary")

    if submit:
        temp_file = "./temp.txt"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        # This is a long document we can split up.
        with open(temp_file) as f:
            alice_in_wonderland = f.read()
    
        texts = text_splitter.create_documents([alice_in_wonderland])
        st.success(f"You have {len(texts)} documents. Displaying the 1st 3 Chunks:")
        st.write(f"Chunk 1,  {len(texts[0].page_content)} characters")
        st.info(texts[0])
        st.write(f"Chunk 2,  {len(texts[1].page_content)} characters")
        st.info(texts[1])
        st.write(f"Chunk 3,  {len(texts[2].page_content)} characters")
        st.info(texts[2])        


# print(texts[0])
# st.write(texts[0])
# print(texts[1])
# st.write(texts[1])