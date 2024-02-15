import streamlit as st
import boto3
from helpers import set_page_config

set_page_config()

from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

st.title("Titan ChatAssistant with Prompt Template")
st.write("""PromptTemplate is responsible for the construction of this input. \
LangChain provides several classes and functions to make constructing and working with prompts easy. \
We will use the default PromptTemplate here.""")

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
    llm=titan_llm, verbose=False, memory=st.session_state.memory)

template ="""System: The following is a friendly conversation between a knowledgeable helpful Assistant and a customer. \
The Assistant is talkative and provides lots of specific details from it's context.

Current conversation:
{history}
User: {input}
Assistant: """

conversation.prompt.template = template

container = st.container(border=True)
container.write(":orange[Template]")
container.code(template, language="markdown")


def form_callback():
	st.session_state.messages = []
	st.session_state.memory.clear()
	del st.session_state.memory

st.sidebar.button(label='Clear Chat History', on_click=form_callback)


# Initialize chat history
if "messages" not in st.session_state.keys():
	st.session_state.messages = [{"role": "Assistant", "content": "How may I assist you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.write(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
	# Add user message to chat history
	st.session_state.messages.append({"role": "User", "content": prompt})
	# Display user message in chat message container
	with st.chat_message("User"):
		st.write(prompt)
	
	# Display Assistant response in chat message container
	with st.chat_message("Assistant"):
		with st.spinner("Thinking..."):
			response = conversation.invoke({'input': prompt})
			#print(response)
			st.write(response['response'])
		st.session_state.messages.append({"role": "Assistant", "content": response['response']})