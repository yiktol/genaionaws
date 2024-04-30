# ------------------------------------------------------------------------
# Streamlit Chat with DynamoDB Memory - Amazon Bedrock and LangChain
# ------------------------------------------------------------------------
import streamlit as st
import boto3
import botocore
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory

# Page title
st.set_page_config(page_title='Chatbot with DynamoDB Memory',
                   	page_icon=":brain:",
	layout="wide",
	initial_sidebar_state="expanded",)

if "session_id" not in st.session_state.keys():
    st.session_state.session_id = str(uuid.uuid4())
    
    
st.subheader("Chatbot with DynamoDB Memory")
st.write("""Amazon DynamoDB can act as the persistent memory to your conversational AI application""")


# ------------------------------------------------------------------------
# Amazon Bedrock Settings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_kwargs = {}

with st.sidebar:
	with st.container(border=True):
		model = st.selectbox('model', ["anthropic.claude-3-sonnet-20240229-v1:0","mistral.mistral-large-2402-v1:0","amazon.titan-tg1-large","meta.llama2-70b-chat-v1"])
		temperature = st.slider('temperature', min_value=0.0,max_value=1.0, value=0.1, step=0.1)
		top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
		max_tokens = st.slider('max_tokens', min_value=50, max_value=4096, value=1024, step=10) 

provider=model.split(":")[0]
match provider:
    case "anthropic":
        model_kwargs =  {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 200,
            "top_p": top_p,
            "stop_sequences": ["\n\nHuman"],
        }
    case "mistral":
        model_kwargs =  {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 200,
            "top_p": top_p,
        }
    case "amazon":
        model_kwargs =  {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }
    case "meta":
        model_kwargs =  {
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    

# model_kwargs =  { 
#     "max_tokens": 2048,
#     "temperature": 0.1,
#     "top_k": 200,
#     "top_p": 0.9,
#     # "stop_sequences": ["\n\nHuman"],
#     # "messages": [{"role": "user", "content": prompt}]
# }
model_id = model

# ------------------------------------------------------------------------
# DynamoDB

TableName="SessionTable"
boto3_session = boto3.Session(region_name="us-east-1")
client = boto3_session.client('dynamodb')
dynamodb = boto3_session.resource("dynamodb")
table = dynamodb.Table(TableName)

# ------------------------------------------------------------------------
# LCEL: chain(prompt | model | output_parser) + RunnableWithMessageHistory

template = [
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
]

prompt = ChatPromptTemplate.from_messages(template)

model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

chain = prompt | model | StrOutputParser()

# DynamoDB Chat Message History
history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="0", boto3_session=boto3_session)

# Chain with History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name="SessionTable", session_id=session_id, boto3_session=boto3_session
    ),
    input_messages_key="question",
    history_messages_key="history",
)

# ------------------------------------------------------------------------
# Streamlit


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear Chat History
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Reset Session
def reset_session():
    st.session_state.session_id = str(uuid.uuid4())
    clear_screen()

# Sidebar
with st.sidebar:
    with st.container(border=True):
        st.write(":orange[Manage DynamoDB Table]")
        if st.button('Create DynamoDB Table'): # Create DynamoDB Table Sidebar Button
            try:
                table = dynamodb.create_table(
                    TableName='SessionTable',
                    KeySchema=[{'AttributeName': 'SessionId', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': 'SessionId', 'AttributeType': 'S'}],
                    BillingMode='PAY_PER_REQUEST'
                )
                table.meta.client.get_waiter('table_exists').wait(TableName='SessionTable')
                st.sidebar.write('DynamoDB table created!')

            except botocore.exceptions.ClientError as error:
                if error.response['Error']['Code'] == 'ResourceInUseException':
                    st.sidebar.info('Table already exists!')
                elif error.response['Error']['Code'] == 'AccessDeniedException':
                    st.sidebar.warning('Access Denied')
                else:
                    raise error
        if st.button('Delete DynamoDB Table'): # Delete DynamoDB Table Sidebar Button
            try:
                response = client.delete_table(
                    TableName='SessionTable'
                )
                st.sidebar.success(f"DynamoDB table is deleted!")
            except client.exceptions.ResourceNotFoundException:
                st.sidebar.info("Table does not exist, nothing to delete")         
            except Exception as e:
                if "AccessDeniedException" in str(e):
                    st.sidebar.warning("Access Denied")
                else:
                    st.sidebar.write(f"Error deleting DynamoDB table: {e}")
    with st.container(border=True):
        st.markdown(':orange[Enable/Disable Streaming]')
        streaming_on = st.toggle('Streaming')
        
    with st.container(border=True):
        st.markdown(':orange[Manage Session]')
        st.button('Clear Screen', on_click=clear_screen)
        st.button('Reset Session', on_click=reset_session)
        st.write('Session ID:', st.session_state.session_id)

# Streamlit Chat Input - User Prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # This is where we configure the session id
    config = {"configurable": {"session_id": st.session_state.session_id}}

    if streaming_on:
        # Chain - Stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            for chunk in chain_with_history.stream({"question": prompt}, config = config):
                full_response += chunk
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Chain - Invoke
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):        
                response = chain_with_history.invoke({"question": prompt}, config = config)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})