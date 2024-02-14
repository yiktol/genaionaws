import streamlit as st
import boto3
from helpers import set_page_config

set_page_config()

from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate

st.title("Titan ChatAssistant with Prompt Template")

bedrock_runtime = boto3.client(
service_name='bedrock-runtime',
region_name='us-east-1', 
)

modelId = "amazon.titan-tg1-large"
titan_llm = Bedrock(model_id=modelId, client=bedrock_runtime)
titan_llm.model_kwargs = {'temperature': 0.5, "maxTokenCount": 700}

template ="""System: The following is a friendly conversation between a knowledgeable helpful Assistant and a customer. \
The Assistant is talkative and provides lots of specific details from it's context.

Current conversation:
{history}
User: {input}
Assistant: """

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

container = st.container(border=True)
container.write(":orange[Template]")
container.code(template, language="markdown")


def form_callback():
	st.session_state.messages = []

st.sidebar.button(label='Clear Chat History', on_click=form_callback)


# Initialize chat history
if "messages" not in st.session_state.keys():
	st.session_state.messages = [{"role": "Assistant", "content": "How may I assist you today?"}]

# Function for generating chat history
def chat_history():
	string_dialogue = ""
	for dict_message in st.session_state.messages:
		#print(dict_message)
		string_dialogue += dict_message['role'] +  " " +dict_message["content"] + "\n"
	return string_dialogue[:-2]


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

	prompt_data = PROMPT.format(input=prompt,history=chat_history())
	
	# Display Assistant response in chat message container
	with st.chat_message("Assistant"):
		with st.spinner("Thinking..."):
			print(prompt_data)
			response = titan_llm(prompt_data)
			st.markdown(response)
		st.session_state.messages.append({"role": "Assistant", "content": response})
	#print(st.session_state.chat_history)