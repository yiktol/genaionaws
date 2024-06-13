import streamlit as st
import utils.helpers as helpers
import utils.stlib as stlib
import logging
import sys
import boto3

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

stlib.set_page_config()

def generate_conversation(bedrock_client,
						  model_id,
						  system_prompts,
						  messages,
						  **params):
	"""
	Sends messages to a model.
	Args:
		bedrock_client: The Boto3 Bedrock runtime client.
		model_id (str): The model ID to use.
		system_prompts (JSON) : The system prompts for the model to use.
		messages (JSON) : The messages to send to the model.

	Returns:
		response (JSON): The conversation that the model generated.

	"""

	logger.info("Generating message with model %s", model_id)

	# Base inference parameters to use.
	inference_config = params
	# Additional inference parameters to use.
	additional_model_fields = {}

	# Send the message.
	response = bedrock_client.converse(
		modelId=model_id,
		messages=messages,
		system=system_prompts,
		inferenceConfig=inference_config,
		additionalModelRequestFields=additional_model_fields
	)

	# Log token usage.
	token_usage = response['usage']
	logger.info("Input tokens: %s", token_usage['inputTokens'])
	logger.info("Output tokens: %s", token_usage['outputTokens'])
	logger.info("Total tokens: %s", token_usage['totalTokens'])
	logger.info("Stop reason: %s", response['stopReason'])

	return response

def main(model_id, params):
	"""
	Entrypoint for Anthropic Claude 3 Sonnet example.
	"""

	logging.basicConfig(level=logging.INFO,
						format="%(levelname)s: %(message)s")

	# model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


 
	with st.container(border=True):
		system_prompt = st.text_area(
			"Enter the system prompt here", value="You are an app that creates playlists for a radio station that plays rock and pop music."
						"Only return song names and the artist.", height=100,)
		prompt = st.text_area("Enter your user prompt here", value="Create a list of 3 pop songs.", height=100,)
		submit = st.button("Submit", type="primary")

	if submit:
		with st.spinner("Thinking..."):		
			# Setup the system prompts and messages to send to the model.
			system_prompts = [{"text": system_prompt}]
			message_1 = {
				"role": "user",
				"content": [{"text": prompt}]
			}
			message_2 = {
				"role": "user",
				"content": [{"text": "Make sure the songs are by artists from the United Kingdom."}]
			}
			messages = []


			try:

				bedrock_client = boto3.client(service_name='bedrock-runtime',region_name = 'us-east-1')

				# Start the conversation with the 1st message.
				messages.append(message_1)
				response = generate_conversation(
					bedrock_client, model_id, system_prompts, messages,**params)

				# Add the response message to the conversation.
				output_message = response['output']['message']
				messages.append(output_message)

				# Continue the conversation with the 2nd message.
				# messages.append(message_2)
				# response = generate_conversation(
				# 	bedrock_client, model_id, system_prompts, messages,**params)

				# output_message = response['output']['message']
				# messages.append(output_message)

				# Show the complete conversation.
				for message in messages:
					st.write(f"Role: {message['role']}")
					for content in message['content']:
						st.write(f"Text: {content['text']}")
					st.write()

			except ClientError as err:
				message = err.response['Error']['Message']
				logger.error("A client error occurred: %s", message)
				st.error(f"A client error occured: {message}")

			else:
				st.write(
					f"Finished generating text with model {model_id}.")


def generate_conversation2(bedrock_client,
						  model_id,
						  input_text,
						  input_image):
	"""
	Sends a message to a model.
	Args:
		bedrock_client: The Boto3 Bedrock runtime client.
		model_id (str): The model ID to use.
		input text : The input message.
		input_image : The input image.

	Returns:
		response (JSON): The conversation that the model generated.

	"""

	logger.info("Generating message with model %s", model_id)

	# Message to send.

	with open(input_image, "rb") as f:
		image = f.read()

	message = {
		"role": "user",
		"content": [
			{
				"text": input_text
			},
			{
					"image": {
						"format": 'png',
						"source": {
							"bytes": image
						}
					}
			}
		]
	}

	messages = [message]

	# Send the message.
	response = bedrock_client.converse(
		modelId=model_id,
		messages=messages
	)

	return response


def main2(model_id):
	"""
	Entrypoint for Anthropic Claude 3 Sonnet example.
	"""

	logging.basicConfig(level=logging.INFO,
						format="%(levelname)s: %(message)s")

	input_image = "images/sg_skyline.jpg"
	st.image(input_image, width=600)
 
	with st.container(border=True):
		input_text = st.text_area("Enter your user prompt here", value="What's in this image?", height=100,)
  		
		submit = st.button("Submit", type="primary", key="Image_button")

	if submit:
		with st.spinner("Thinking..."):
			try:

				bedrock_client = boto3.client(service_name="bedrock-runtime",region_name = 'us-east-1')

				response = generate_conversation2(
					bedrock_client, model_id, input_text, input_image)

				output_message = response['output']['message']

				st.write(f"Role: {output_message['role']}")

				for content in output_message['content']:
					st.write(f"Text: {content['text']}")

				token_usage = response['usage']
				st.write(f"Input tokens:  {token_usage['inputTokens']}")
				st.write(f"Output tokens:  {token_usage['outputTokens']}")
				st.write(f"Total tokens:  {token_usage['totalTokens']}")
				st.write(f"Stop reason: {response['stopReason']}")

			except ClientError as err:
				message = err.response['Error']['Message']
				logger.error("A client error occurred: %s", message)
				st.error(f"A client error occured: {message}")

			else:
				st.write(
					f"Finished generating text with model {model_id}.")



def stream_conversation(bedrock_client,
					model_id,
					messages,
					system_prompts,
					inference_config,
					additional_model_fields):
	"""
	Sends messages to a model and streams the response.
	Args:
		bedrock_client: The Boto3 Bedrock runtime client.
		model_id (str): The model ID to use.
		messages (JSON) : The messages to send.
		system_prompts (JSON) : The system prompts to send.
		inference_config (JSON) : The inference configuration to use.
		additional_model_fields (JSON) : Additional model fields to use.

	Returns:
		Nothing.

	"""

	logger.info("Streaming messages with model %s", model_id)

	response = bedrock_client.converse_stream(
		modelId=model_id,
		messages=messages,
		system=system_prompts,
		inferenceConfig=inference_config,
		additionalModelRequestFields=additional_model_fields
	)

	stream = response.get('stream')
	if stream:
		placeholder = st.empty()
		full_response = ''
		for event in stream:

			if 'messageStart' in event:
				st.write(f"\nRole: {event['messageStart']['role']}")

			# if 'contentBlockDelta' in event:
			# 	st.write(event['contentBlockDelta']['delta']['text'], end="")
   

			if 'contentBlockDelta' in event:
				chunk = event['contentBlockDelta']
				part = chunk['delta']['text']
				full_response += part
				placeholder.info(full_response)
			placeholder.info(full_response)

			if 'messageStop' in event:
				st.write(f"\nStop reason: {event['messageStop']['stopReason']}")

			if 'metadata' in event:
				metadata = event['metadata']
				if 'usage' in metadata:
					st.write("\nToken usage")
					st.write(f"Input tokens: {metadata['usage']['inputTokens']}")
					st.write(
						f":Output tokens: {metadata['usage']['outputTokens']}")
					st.write(f":Total tokens: {metadata['usage']['totalTokens']}")
				if 'metrics' in event['metadata']:
					st.write(
						f"Latency: {metadata['metrics']['latencyMs']} milliseconds")


def main_stream(model_id, params):
	"""
	Entrypoint for streaming message API response example.
	"""

	logging.basicConfig(level=logging.INFO,
						format="%(levelname)s: %(message)s")
	
	with st.container(border=True):
		system_prompt = st.text_area(
			"Enter the system prompt here", value="You are an app that creates playlists for a radio station that plays rock and pop music."
						"Only return song names and the artist.", height=100, key='streaming_system')
		input_text = st.text_area("Enter your user prompt here", value="Create a list of 3 pop songs.", height=100,key='streaming_input')
		submit = st.button("Submit", type="primary", key='Streaming')

	if submit:
		with st.spinner("Thinking..."):	

			message = {
				"role": "user",
				"content": [{"text": input_text}]
			}
			messages = [message]
			
			# System prompts.
			system_prompts = [{"text" : system_prompt}]

			# Base inference parameters.
			inference_config = params
			# Additional model inference parameters.
			additional_model_fields = {}

			try:
				bedrock_client = boto3.client(service_name='bedrock-runtime',region_name = 'us-east-1')

				stream_conversation(bedrock_client, model_id, messages,
								system_prompts, inference_config, additional_model_fields)

			except ClientError as err:
				message = err.response['Error']['Message']
				logger.error("A client error occurred: %s", message)
				st.error("A client error occured: " +
					format(message))

			else:
				st.write(
					f"Finished streaming messages with model {model_id}.")






MODEL_IDS = [
	"anthropic.claude-3-sonnet-20240229-v1:0",
	"anthropic.claude-3-haiku-20240307-v1:0",
	"cohere.command-r-plus-v1:0",
	"cohere.command-r-v1:0",
	"meta.llama3-70b-instruct-v1:0",
	"meta.llama3-8b-instruct-v1:0",
	"mistral.mistral-large-2402-v1:0",
	"mistral.mixtral-8x7b-instruct-v0:1",
	"mistral.mistral-7b-instruct-v0:2",
	"mistral.mistral-small-2402-v1:0"
	]

def tune_parameters():
	temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
	topP = st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
	max_tokens = st.number_input('max_tokens',min_value = 50, max_value = 4096, value = 1024, step = 1)
	params = {
		"temperature":temperature, 
		"topP":topP,
		"maxTokens":max_tokens,
		}

	return params

text, parameters = st.columns([0.7, 0.3])

with parameters:

	with st.container(border=True):
		model = st.selectbox('model', MODEL_IDS)
		params = tune_parameters()
		
with text:
	st.title('Converse API in Amazon Bedrock')
	st.write("""The Converse or ConverseStream API is a unified structured text API action that allows you simplifying the invocations to Bedrock LLMs, using a universal syntax and message structured prompts for any of the supported model providers.""")

	tabs = st.tabs(['Conversation with text message', 'Conversation with image','Conversation streaming'])

	with tabs[0]:
		with st.expander("See Code"):
			with open("templates/converse1.py") as my_file:
				st.code(my_file.read(), language="python")
			

		main(model_id=model,params=params)
  
	with tabs[1]:
		with st.expander("See Code"):
			with open("templates/converse2.py") as my_file:
				st.code(my_file.read(), language="python")

		main2(model_id=model)
  
	with tabs[2]:
		with st.expander("See Code"):
			with open("templates/converse3.py") as my_file:
				st.code(my_file.read(), language="python")

		main_stream(model_id=model, params=params)

