import streamlit as st
import boto3
from helpers import set_page_config
from langchain_community.llms import Bedrock


set_page_config()

bedrock_runtime = boto3.client(
service_name='bedrock-runtime',
region_name='us-east-1', 
)

modelId = "amazon.titan-tg1-large"
titan_llm = Bedrock(model_id=modelId, client=bedrock_runtime)
titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}


# Initialize chat history
if "messages" not in st.session_state.keys():
	st.session_state.messages = [{"role": "Assistant", "content": "How may I assist you today?"}]

st.title("Titan Chatbot without Context")


def form_callback():
    st.session_state.messages = []

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "User", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)

 # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = titan_llm(prompt)
            st.markdown(stream)
        #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})

