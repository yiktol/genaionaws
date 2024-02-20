import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from helpers import bedrock_runtime_client, set_page_config

bedrock = bedrock_runtime_client()
set_page_config()

def form_callback():
    st.session_state.messages = []
    st.session_state.memory.clear()
    del st.session_state.memory

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)

st.header("Memory")
st.markdown("""Most LLM applications have a conversational interface. \
An essential component of a conversation is being able to refer to information introduced earlier in the conversation. \
At bare minimum, a conversational system should be able to access some window of past messages directly. \
A more complex system will need to have a world model that it is constantly updating, which allows it to do things like maintain information about entities and their relationships.
            """)

st.subheader(":orange[Memory in LLMChain]")
st.markdown("""There are many different types of memory. Each has their own parameters, their own return types, and is useful in different scenarios.""")

expander = st.expander("See code")
expander.code("""from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_community.chat_models import BedrockChat

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = BedrockChat(client=bedrock,model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

chat_llm_chain.predict(human_input="Hi there my friend")
    """,language="python")

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

# Initialize chat history
if "messages" not in st.session_state.keys():
	st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = BedrockChat(client=bedrock,model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True,
    memory=st.session_state.memory,
)


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
            stream = chat_llm_chain.predict(human_input=prompt)
            st.markdown(stream)
        #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})