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
from langchain_community.chat_models import BedrockChat
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


model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
    # "messages": [{"role": "user", "content": prompt}]
}
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

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

model = BedrockChat(
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

# Sidebar
with st.sidebar:
    st.subheader('With DynamoDB Memory :brain:')
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
                st.sidebar.write('Table already exists!')
            else:
                raise error
    if st.button('Delete DynamoDB Table'): # Delete DynamoDB Table Sidebar Button
        try:
            response = client.delete_table(
                TableName='SessionTable'
            )
            st.sidebar.write(f"DynamoDB table is deleted!")
        except client.exceptions.ResourceNotFoundException:
            st.sidebar.write("Table does not exist, nothing to delete")
        except Exception as e:
            st.sidebar.write(f"Error deleting DynamoDB table: {e}")
    streaming_on = st.toggle('Streaming')
    st.button('Clear Screen', on_click=clear_screen)

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
        with st.spinner("Thinking..."):        
            with st.chat_message("assistant"):
                response = chain_with_history.invoke({"question": prompt}, config = config)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})