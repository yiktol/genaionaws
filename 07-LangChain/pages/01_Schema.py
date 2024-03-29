import streamlit as st
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from helpers import bedrock_runtime_client, set_page_config

set_page_config()
bedrock = bedrock_runtime_client()

st.header("LangChain Schema")
st.markdown("Schema covers the basic data types and schemas that are used throughout the codebase. It comprises four primary elements: Text, ChatMessages, Examples, and Document.")

st.subheader(":orange[Text]")
st.markdown("Text is a string that represents the input to the model.")

with st.form("form1"):
    prompt = st.text_input("Prompt:", value="What is the capital of France?")
    submit = st.form_submit_button("Submit",type="primary")

st.code(f"""
        prompt = \'{prompt}\'
        
        """,language="python")

st.subheader(":orange[ChatMessages]")
st.markdown("""Like text, but specified with a message type (System, Human, AI)\n
- System - A Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
- Human - A Message from a human.
- AI - A Message from an AI.""")        

expander = st.expander("See code")

expander.code("""from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import BedrockChat

chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})           

messages = [
    SystemMessage(content="You are a nice AI bot that translate English to Spanish language"),
    HumanMessage(content="I love programming."),
    AIMessage(content="I am an AI bot that translate English to Spanish language")
]
chat(messages)""")

with st.form("form2"):
    prompt2 = st.text_input("Prompt:", value="I love programming")
    submit2 = st.form_submit_button("Submit",type="primary")

    chat = BedrockChat(client=bedrock,model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})

    messages = [
        SystemMessage(content="You are a nice AI bot that translate English to Spanish language"),
        HumanMessage(content=prompt2),
        AIMessage(content="I am an AI bot that translate English to Spanish language")
    ]
    if submit2:
        response = chat.invoke(messages)
        # print(response)
        st.write("Answer:")
        st.info(f"{response.content}")
    
    
st.subheader(":orange[Documents]")
st.markdown("Documents are the basic unit of information in LangChain. They consist of a list of chunks, each of which is a string.")

expander2 = st.expander("See code")

expander2.code("""from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="bedrock_faqs.csv")

Schema = Document(page_content="This is my document.",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "The LangChain Papers",
             'my_document_create_time' : 1680013019
         })

document = loader.load()     
print(document)""",language="python")

from langchain_community.document_loaders.csv_loader import CSVLoader

with st.form("form3"):
    prompt3 = st.text_input("File:", value="bedrock_faqs.csv",disabled=True)
    submit3 = st.form_submit_button("Submit",type="primary")

    if submit3:
        loader = CSVLoader(file_path="bedrock_faqs.csv")
        data = loader.load()
        st.write("Answer:")
        st.write(data[:5])