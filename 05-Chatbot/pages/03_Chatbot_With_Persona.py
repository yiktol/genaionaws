import streamlit as st
from utils.helpers import set_page_config, bedrock_runtime_client
from langchain.chains import LLMChain
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


set_page_config()

def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Reset Session', on_click=reset_session)

def form_callback():
    st.session_state.messages = []
    st.session_state.memory.clear()
    del st.session_state.memory

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "AI", "content": "How may I assist you today?"}]

if "memory" not in st.session_state:
     st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)


st.title("Chatbot with Persona")
st.write("AI assistant will play the role of a career coach. Role Play Dialogue requires user message to be set in before starting the chat. ConversationBufferMemory is used to pre-populate the dialog.")

bedrock = bedrock_runtime_client()

modelId = "meta.llama2-13b-chat-v1"
llm = BedrockChat(model_id=modelId, client=bedrock)

with st.form(key ='Form1'):
    user_message= st.text_input(':orange[User Message:]',value='You will be acting as a career coach. Your goal is to give career advice to users. You say \'I don\'t know\' if the message is not related to career questions.')
    ai_message= st.text_input(':orange[AI Message:]', value='I am career coach and give career advice.')
    submitted = st.form_submit_button(label = 'Set Persona', type='primary') 
            

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=user_message
        ), 
        AIMessage(content=ai_message),
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



# Initialize chat history

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

