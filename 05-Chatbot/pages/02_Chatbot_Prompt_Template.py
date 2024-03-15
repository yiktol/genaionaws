import streamlit as st
import boto3
from utils.helpers import set_page_config, bedrock_runtime_client
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import (ChatPromptTemplate, 
                               SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate, 
                               MessagesPlaceholder,)
set_page_config()
st.title("Chat with Prompt Template")
st.write("""PromptTemplate is responsible for the construction of this input. \
LangChain provides several classes and functions to make constructing and working with prompts easy. \
We will use the default PromptTemplate here.""")

bedrock = bedrock_runtime_client()

modelId = "meta.llama2-13b-chat-v1"
llm = BedrockChat(model_id=modelId, client=bedrock)


if "memory" not in st.session_state:
     st.session_state.memory = ConversationBufferMemory(return_messages=True)
     st.session_state.memory.human_prefix = "Human"
     st.session_state.memory.ai_prefix = "AI"
     st.session_state.memory.memory_key = "history"

template ="""_System: The following is a friendly conversation between a knowledgeable helpful Assistant and a customer. \
The Assistant is talkative and provides lots of specific details from it's context._

_Current conversation:_

_:blue[{history}]_

_Human: :blue[{input}]_

_AI:_ """


prompt_template = ChatPromptTemplate(
    messages=[
		SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a knowledgeable helpful AI and a customer."),
		MessagesPlaceholder(variable_name="history"),
		HumanMessagePromptTemplate.from_template("{input}"),
		]
	)

conversation = LLMChain(
    llm=llm, verbose=True, memory=st.session_state.memory, prompt=prompt_template)

#conversation.prompt.template = template

container = st.container(border=True)
container.write(":orange[Template]")
container.markdown(template)

def form_callback():
	st.session_state.messages = []
	st.session_state.memory.clear()
	del st.session_state.memory

st.sidebar.button(label='Clear Chat History', on_click=form_callback)

def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Reset Session', on_click=reset_session)

# Initialize chat history
if "messages" not in st.session_state.keys():
	st.session_state.messages = [{"role": "AI", "content": "How may I assist you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.write(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
	# Add user message to chat history
	st.session_state.messages.append({"role": "Human", "content": prompt})
	# Display user message in chat message container
	with st.chat_message("Human"):
		st.write(prompt)
	
	# Display Assistant response in chat message container
	with st.chat_message("AI"):
		with st.spinner("Thinking..."):
			response = conversation({'input': prompt})
			#print(response)
			st.write(response['text'])
		st.session_state.messages.append({"role": "AI", "content": response['text']})