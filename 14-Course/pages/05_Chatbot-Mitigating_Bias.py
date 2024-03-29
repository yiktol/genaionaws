import streamlit as st
import boto3
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import (ChatPromptTemplate, 
							   SystemMessagePromptTemplate, 
							   HumanMessagePromptTemplate, 
							   MessagesPlaceholder,)
import utils.sdxl as sdxl

st.set_page_config(
	page_title="Mitigating Bias - Chatbot",
	page_icon=":rocket:",
	layout="wide",
	initial_sidebar_state="expanded",
)
st.title("Chat - Mitigating Bias")

bedrock = boto3.client(service_name='bedrock-runtime',region_name='us-east-1' )

modelId = "anthropic.claude-v2:1"
llm = BedrockChat(model_id=modelId, client=bedrock)


def generate_image(prompt_data):
	with st.spinner("Generating Image..."):
		generated_image = sdxl.get_image_from_model(
						prompt = prompt_data, 
						negative_prompt = "bias,discriminatory,poorly rendered,poor background details,poorly drawn feature,disfigured features",
						model="stability.stable-diffusion-xl-v1",
						height = 1024,
						width = 1024,
						cfg_scale = 5, 
						seed = 123456789,
						steps = 20,
						style_preset = "photographic"
						
					)
	st.image(generated_image)



if "memory" not in st.session_state:
	 st.session_state.memory = ConversationBufferMemory(return_messages=True)
	 st.session_state.memory.human_prefix = "Human"
	 st.session_state.memory.ai_prefix = "Assistant"
	 st.session_state.memory.memory_key = "history"

template ="""{history}

Human: {input}

Assistant:"""


prompt_template = ChatPromptTemplate(
	messages=[
		SystemMessagePromptTemplate.from_template("""You are a prompt generator, who generates prompts for text to image models based on a user question. \
You and the image AI are not biased and do not discriminate against certain groups of people. If you detect bias in the question, ask relevant questions based on gender, race and color before
generating the prompt. If you don't know the answer to a question, truthfully say you don't know. If the image generation question includes human beings, ask all of the following questions with
examples before generating the image prompt:

<questions>
- What is the gender of the subject in the picture? (e.g. male, female, transgender etc.)
- What is the color of the subject in the image? (e.g. white, black, or brown etc.)
- What is the race of the subject in the image? (e.g. African-american, latino, indian, korean,
  chineese, asian, etc.)
</questions>

When you are ready to generate the image prompt, return it in <imageprompt></imageprompt> XML tags.

Assistant:OK, I understand
"""),
		MessagesPlaceholder(variable_name="history"),
		HumanMessagePromptTemplate.from_template("{input}"),
		]
	)

conversation = LLMChain(
	llm=llm, verbose=True, memory=st.session_state.memory, prompt=prompt_template)

#conversation.prompt.template = template

with st.expander("See Template"):
	st.markdown("""
Human: You are a prompt generator, who generates prompts for text to image models based on a user question.
You and the image AI are not biased and do not discriminate against certain groups of people.
If you detect bias in the question, ask relevant questions based on gender, race and color before
generating the prompt. If you don't know the answer to a question, truthfully say you don't know.
If the image generation question includes human beings, ask all of the following questions with
examples before generating the image prompt:

<questions>

- What is the gender of the subject in the picture? (e.g. male, female, transgender etc.)
- What is the color of the subject in the image? (e.g. white, black, or brown etc.)
- What is the race of the subject in the image? (e.g. African-american, latino, indian, korean,
  chineese, asian, etc.)\n
</questions>\n

When you are ready to generate the image prompt, return it in <imageprompt></imageprompt> XML tags.

Assistant: OK, I understand

{history}

Human: {input}

Assistant:
 """)

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
	st.session_state.messages = [{"role": "Assistant", "content": "How may I assist you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.write(message["content"])

# Accept user input
if prompt := st.chat_input("Create a photo of a doctor."):
	# Add user message to chat history
	st.session_state.messages.append({"role": "Human", "content": prompt})
	# Display user message in chat message container
	with st.chat_message("Human"):
		st.write(prompt)
	
	# Display Assistant response in chat message container
	with st.chat_message("Assistant"):
		with st.spinner("Thinking..."):
			response = conversation({'input': prompt})
			if "<imageprompt>" in response['text']:
				ix_prompt_start = response['text'].find("<imageprompt>") + len("<imageprompt>")
				ix_prompt_end = response['text'].find("</imageprompt>", ix_prompt_start)
				img_prompt = response['text'][ix_prompt_start:ix_prompt_end].strip()
				st.write(response['text'])
				generate_image(img_prompt)
			else:
			#print(response)
				st.write(response['text'])
		st.session_state.messages.append({"role": "Assistant", "content": response['text']})