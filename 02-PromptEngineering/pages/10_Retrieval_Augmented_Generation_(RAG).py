# main.py

import os
import boto3
import streamlit as st
from llama_index.core import ( 
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.settings import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding, Models

# ------------------------------------------------------------------------
# LlamaIndex - Amazon Bedrock

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

llm = Bedrock(model = "anthropic.claude-v2", client = bedrock)
embed_model = BedrockEmbedding(model = "amazon.titan-embed-text-v1", client = bedrock)

Settings.llm = llm
Settings.embed_model = embed_model

# ------------------------------------------------------------------------
# Streamlit

# Page title
st.set_page_config(page_title='RAG',
                   page_icon=":brain:",
	layout="wide",
	initial_sidebar_state="expanded",)

# Clear Chat History fuction
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.title('LlamaIndex 🦙')
    st.subheader('Q&A over you data 📂')
    st.markdown('[Amazon Bedrock](https://aws.amazon.com/bedrock/) - The easiest way to build and scale generative AI applications with foundation models')
    st.divider()
    streaming_on = st.toggle('Streaming')
    st.button('Clear Screen', on_click=clear_screen)

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text = "Loading and indexing your data. This may take a while..."):
        PERSIST_DIR = "storage"
        # check if storage already exists
        if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
            documents = SimpleDirectoryReader(input_dir="data", recursive=True).load_data()
            index = VectorStoreIndex.from_documents(documents)
            # persistent storage 
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
        return index

# Create Index
index = load_data()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input - User Prompt 
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if streaming_on:
        # Query Engine - Streaming
        query_engine = index.as_query_engine(streaming=True)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            streaming_response = query_engine.query(prompt)
            for chunk in streaming_response.response_gen:
                full_response += chunk
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        # Query Engine - Query
        query_engine = index.as_query_engine()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(prompt)
                st.write(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})