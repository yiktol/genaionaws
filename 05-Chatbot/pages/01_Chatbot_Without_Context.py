import streamlit as st
import boto3
from helpers import set_page_config
from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


set_page_config()

bedrock_runtime = boto3.client(
service_name='bedrock-runtime',
region_name='us-east-1', 
)

modelId = "amazon.titan-tg1-large"
titan_llm = Bedrock(model_id=modelId, client=bedrock_runtime)
titan_llm.model_kwargs = {'temperature': 0.1, "maxTokenCount": 700}


if "memory" not in st.session_state:
     st.session_state.memory = ConversationBufferMemory(return_messages=True)
     st.session_state.memory.human_prefix = "User"
     st.session_state.memory.ai_prefix = "assistant"
     st.session_state.memory.memory_key = "history"

conversation = ConversationChain(
    llm=titan_llm, verbose=False, memory=st.session_state.memory
)
conversation.prompt.template = """System: The following is a friendly conversation between a knowledgeable helpful career coach and a customer. \
The career coach is talkative and provides lots of specific details from it's context.\
\n\nCurrent conversation:\n{history}\nUser: {input}\nBot:"""



# Initialize chat history
if "messages" not in st.session_state.keys():
	st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.title("Titan Chatbot without Context")
st.write("""Using CoversationChain from LangChain to start the conversation
Chatbots needs to remember the previous interactions. Conversational memory allows us to do that. \
There are several ways that we can implement conversational memory. In the context of LangChain, they are all built on top of the ConversationChain.

Note: The model outputs are non-deterministic""")


def form_callback():
    st.session_state.messages = []
    del st.session_state.memory

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
            stream = conversation.invoke({'input': prompt})
            st.markdown(stream['response'])
        #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream['response']})
    
    print(st.session_state.memory.load_memory_variables({}))

