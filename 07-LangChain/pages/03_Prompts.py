import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

from helpers import bedrock_runtime_client, set_page_config

bedrock = bedrock_runtime_client()

llm = Bedrock(
    client=bedrock, model_id="amazon.titan-text-express-v1"
)

set_page_config()


st.header("Prompts")
st.markdown("""A prompt for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output, such as answering questions, completing sentences, or engaging in a conversation.
            """)

st.subheader(":orange[Prompt]")
st.markdown("Text is a string that represents the input to the model.")

prompt = """I want to become a Data Engineer.

How can I achieve this?
"""

expander = st.expander("See code")
expander.code(f"""from langchain_community.llms import Bedrock

prompt = \"""{prompt}\"""

llm = Bedrock(
    client=bedrock, model_id="amazon.titan-text-express-v1"
)

llm.invoke(prompt)""",language="python")


with st.form("form1"):
    prompt = st.text_area("Prompt:", value=prompt )
    submit = st.form_submit_button("Submit",type="primary")

    if submit:
        #print(response)
        st.write("Answer:")
        with st.spinner("AI Thinking..."):
            response = llm.invoke(prompt)
            st.info(response)
        
        
st.subheader(":orange[Prompt Template]")
st.markdown("Prompt templates are predefined recipes for generating prompts for language models. A template may include instructions, few-shot examples, and specific context and questions appropriate for a given task.")

expander2 = st.expander("See code")
expander2.code(f"""from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {{adjective}} joke about {{content}}."
)
prompt_template.format(adjective="funny", content="chickens")""",language="python")




prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", content="chickens")

with st.form("form2"):
    adjective = st.text_input("Adjective:", value="funny")
    content = st.text_input("Content:", value="chickens")
    submit2 = st.form_submit_button("Submit",type="primary")

    if submit2:
        response = prompt_template.format(adjective=adjective, content=content)
        st.write("Prompt:")
        st.info(response)
        st.write("Answer:")
        with st.spinner("AI Thinking..."):
            
            st.success(llm.invoke(response))
        
st.subheader(":orange[Chat Prompt Template]")
st.markdown("The prompt to chat models is a list of chat messages.")

expander3 = st.expander("See code")
expander3.code(f"""from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{{text}}"),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
print(messages)""",language="python")


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

with st.form("form3"):
    prompt = st.text_area("Prompt:", value="I don't like eating tasty things" ) 
    submit3 = st.form_submit_button("Submit",type="primary")

    if submit3:
        messages = chat_template.format_messages(text=prompt)
        st.write("Prompt:")
        st.info(messages)
        st.write("Answer:")
        with st.spinner("Bot Thinking..."):
            st.success(llm.invoke(messages))

