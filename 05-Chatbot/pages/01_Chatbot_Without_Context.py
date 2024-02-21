import streamlit as st
import boto3
from helpers import set_page_config, bedrock_runtime_client
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


set_page_config()

bedrock = bedrock_runtime_client()

def form_callback():
    st.session_state.messages = []
    st.session_state.memory.clear()
    del st.session_state.memory

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

llm = BedrockChat(client=bedrock,model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})


prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="The following is a friendly conversation between a knowledgeable helpful career coach and a customer. \
The career coach is talkative and provides lots of specific details from it's context."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)


chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True,
    memory=st.session_state.memory,
)

st.title("Chatbot without Context")
st.write("""Using CoversationChain from LangChain to start the conversation
Chatbots needs to remember the previous interactions. Conversational memory allows us to do that. \
There are several ways that we can implement conversational memory. In the context of LangChain, they are all built on top of the ConversationChain.

Note: The model outputs are non-deterministic""")


# Initialize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
 
c1 = st.container(height=600)
c2 = st.container(height=110)
   
with c1:  
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Accept user input
c2.write('Say something')
if prompt := c2.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "User", "content": prompt})
    # Display user message in chat message container
    with c1.chat_message("User"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with c1.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = chat_llm_chain.predict(human_input=prompt)
            st.markdown(stream)
        #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})
    

