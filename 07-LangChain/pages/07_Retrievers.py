import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from helpers import bedrock_runtime_client, set_page_config


set_page_config()
bedrock = bedrock_runtime_client()

embeddings = BedrockEmbeddings(
    client=bedrock, region_name="us-east-1"
)


st.header("Retrievers")
st.markdown("""A retriever is an interface that returns documents given an unstructured query. \
It is more general than a vector store. A retriever does not need to be able to store documents, only to return (or retrieve) them. \
Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well.
            """)

st.subheader(":orange[Vector store-backed retriever]")
st.markdown("""A vector store retriever is a retriever that uses a vector store to retrieve documents. \
It is a lightweight wrapper around the vector store class to make it conform to the retriever interface. \
It uses the search methods implemented by a vector store, like similarity search and MMR, to query the texts in the vector store. \
[Sample TXT file](https://training.yikyakyuk.com/genai/docs/alice.txt)""")

expander = st.expander("See code")
expander.code("""from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings

loader = TextLoader("alice.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = BedrockEmbeddings(client=bedrock, region_name="us-east-1")

db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

docs = retriever.get_relevant_documents("who is alice?")
    """,language="python")


with st.form("form1"):
    uploaded_file = st.file_uploader(":orange[Choose a TXT file:]", type=['txt'])
    query = st.text_area(":orange[Query:]", value="who is alice?", disabled=False)
    submit = st.form_submit_button("Load TXT",type="primary")

    if submit:
        with st.spinner("Loading TXT...."):
            temp_file = "./temp.txt"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = TextLoader(temp_file)
            documents = loader.load()
            st.success("Done!, Loading")

        with st.spinner("Splitting TXT...."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            st.write(f"Chunk 1,  {len(texts[0].page_content)} characters, You have {len(texts)} documents")
            st.info(texts[0])
            st.success("Done!, Splitting")

        with st.spinner("Embedding TXT...."):
            db = FAISS.from_documents(texts, embeddings)
            retriever = db.as_retriever()
            st.write(f"Retriever: {retriever}")
            st.success("Done!, Embedding")

        with st.spinner("Querying...."):           
            docs = retriever.get_relevant_documents(query)
            st.write(f"Found {len(docs)} documents")
            st.info("\n\n".join([x.page_content for x in docs]))
            st.success("Done!, Querying")