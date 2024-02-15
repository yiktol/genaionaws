import streamlit as st
import boto3
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

st.title("Titan Chatbot with Persona")
st.write("AI assistant will play the role of a career coach. Role Play Dialogue requires user message to be set in before starting the chat. ConversationBufferMemory is used to pre-populate the dialog.")

bedrock_runtime = boto3.client(
service_name='bedrock-runtime',
region_name='us-east-1', 
)

modelId = "amazon.titan-tg1-large"
titan_llm = Bedrock(model_id=modelId, client=bedrock_runtime)
titan_llm.model_kwargs = {'temperature': 0, "maxTokenCount": 700}


with st.form(key ='Form1'):
    user_message= st.text_input('User Message:',value='You will be acting as a career coach. Your goal is to give career advice to users')
    ai_message= st.text_input('AI Message:', value='I am career coach and give career advice')
    submitted = st.form_submit_button(label = 'Set Personality') 
            
if "memory" not in st.session_state:
     st.session_state.memory = ConversationBufferMemory(return_messages=True)
     st.session_state.memory.chat_memory.add_user_message(user_message)
     st.session_state.memory.chat_memory.add_ai_message(ai_message)
     st.session_state.memory.human_prefix = "User"
     st.session_state.memory.ai_prefix = "assistant"

# memory = ConversationBufferMemory()
# memory.chat_memory.add_user_message(user_message)
# memory.chat_memory.add_ai_message(ai_message)
# memory.human_prefix = "User"
# memory.ai_prefix = "assistant"

conversation = ConversationChain(
    llm=titan_llm, verbose=False, memory=st.session_state.memory
)
conversation.prompt.template = """System: The following is a friendly conversation between a knowledgeable helpful career coach and a customer. The career coach is talkative and provides lots of specific details from it's context.\n\nCurrent conversation:\n{history}\nUser: {input}\nBot:"""


def form_callback():
    st.session_state.messages = []
    st.session_state.memory.clear()
    del st.session_state.memory

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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

