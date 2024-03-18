import streamlit as st #all streamlit commands will be available through the "st" alias
from utils.helpers import set_page_config
import uuid, json
import boto3


set_page_config()

client = boto3.client("bedrock-agent-runtime", region_name='us-east-1')


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid1())


def form_callback():
    st.session_state.chat_history.clear()

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)

def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Reset Session', on_click=reset_session)

input_text:str = "I am looking to buy running shoes?" # replace this with a prompt relevant to your agent
agent_id:str = 'W789VIBDQ3' # note this from the agent console on Bedrock
agent_alias_id:str = 'TSTALIASID' # fixed for draft version of the agent
session_id:str = st.session_state.session_id # random identifier
enable_trace:bool = False

def get_chat_response():
    
    response = client.invoke_agent(inputText=input_text,
                                agentId=agent_id,
                                agentAliasId=agent_alias_id,
                                sessionId=session_id,
                                enableTrace=enable_trace
                                )
    
    return response


def output_response(response):
    event_stream = response['completion']
    try:
        for event in event_stream:        
            if 'chunk' in event:
                data = event['chunk']['bytes']
                #st.info(f"{data.decode('utf8')}") 
                end_event_received = True
                # End event indicates that the request finished successfully
            elif 'trace' in event:
                st.info(json.dumps(event['trace'], indent=2))
            else:
                raise Exception("unexpected event.", event)
    except Exception as e:
        raise Exception("unexpected event.", e)
    return data.decode('utf8')



st.subheader("Bedrock Agent")
st.write("Customer Retail Assistant - Shoe Department")

with st.expander(":orange[See Architeture]"):
    st.image("images/91-agents-arch-diagram.png")
with st.expander(":orange[See Workflow]"):
    st.image("images/91-sequence-flow-agent.png")

if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [{"role": "assistant", "text": "Hi, I'm a Customer Reatail Assistant. How may I assist you today?"}]


#Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: #loop through the chat history
    with st.chat_message(message["role"]): #renders a chat line for the given role, containing everything in the with block
        st.markdown(message["text"]) #display the chat content



input_text = st.chat_input("Chat with your bot here") #display a chat input box

if input_text: #run the code in this if block after the user submits a chat message
    
    with st.chat_message("user"): #display a user chat message
        st.markdown(input_text) #renders the user's latest message
    
    st.session_state.chat_history.append({"role":"user", "text":input_text}) #append the user's latest message to the chat history
    
    with st.chat_message("assistant"): #display a bot chat message
        with st.spinner("Thinking..."):
            chat_response = get_chat_response() #call the model through the supporting library
            response =  output_response(chat_response)
            st.markdown(response) #display bot's latest response
        
            st.session_state.chat_history.append({"role":"assistant", "text":response}) #append the bot's latest message to the chat history
        
