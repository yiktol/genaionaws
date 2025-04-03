import streamlit as st #all streamlit commands will be available through the "st" alias
import utils.chatbot_lib as glib #reference to local lib script
from utils.helpers import set_page_config

set_page_config()

def form_callback():
    # st.session_state.chat_history = []
    st.session_state.chat_history.clear()

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)

def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Reset Session', on_click=reset_session)


st.title("Chatbot without Context")
st.write("""Using CoversationChain from LangChain to start the conversation
Chatbots needs to remember the previous interactions. Conversational memory allows us to do that. \
There are several ways that we can implement conversational memory. In the context of LangChain, they are all built on top of the ConversationChain.

Note: The model outputs are non-deterministic""")


if 'memory' not in st.session_state: #see if the memory hasn't been created yet
    st.session_state.memory = glib.get_memory() #initialize the memory


if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [{"role": "AI", "text": "How may I assist you today?"}]


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
            chat_response = glib.get_chat_response(input_text=input_text, memory=st.session_state.memory) #call the model through the supporting library
            st.markdown(chat_response) #display bot's latest response
        
        st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) #append the bot's latest message to the chat history
        
