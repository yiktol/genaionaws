import streamlit as st
from operator import itemgetter
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import ChatPromptTemplate
from helpers import bedrock_runtime_client, set_page_config

bedrock = bedrock_runtime_client()
set_page_config()

st.header("Chains")
st.markdown("""Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. \
The primary supported way to do this is with LCEL.
            """)

st.subheader(":orange[Prompt + LLM]")
st.markdown("""The most common and valuable composition is taking:\n
_PromptTemplate / ChatPromptTemplate -> LLM / ChatModel -> OutputParser_\n
Almost any other chains you build will use this building block.""")

expander = st.expander("See code")
expander.code("""from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Bedrock

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = Bedrock(client=bedrock, model_id="amazon.titan-text-express-v1")

chain = prompt | model 

response = chain.invoke({"topic": "cake"})
print(response)

    """,language="python")

template = "Tell me a joke about {topic}"

prompt = ChatPromptTemplate.from_template(template)
model = Bedrock(client=bedrock, model_id="anthropic.claude-v2")


with st.form("form1"):
    st.markdown(f":orange[Template:] _:blue[{template}]_" )
    topic = st.text_input("topic:",value="cake")
    submit = st.form_submit_button("Submit",type="primary")

    if submit:
        with st.spinner(f"Running..."):
            chain = prompt | model
            response = chain.invoke({"topic": topic})
            st.write("Answer:")
            st.info(response)
        
st.subheader(":orange[Multiple chains]")
st.markdown("""Runnables can easily be used to string together multiple Chains""")

expander = st.expander("See code")
expander.code("""from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Bedrock

prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

model = Bedrock(client=bedrock, model_id="anthropic.claude-v2")

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

chain2.invoke({"person": "obama", "language": "spanish"})

    """,language="python")


from operator import itemgetter

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt1_template = "what is the city {person} is from?"
prompt2_template = "what country is the city {city} in? respond in {language}"

prompt1 = ChatPromptTemplate.from_template(prompt1_template)
prompt2 = ChatPromptTemplate.from_template(prompt2_template)

model = Bedrock(client=bedrock, model_id="anthropic.claude-v2")

with st.form("form2"):
    st.markdown(f":orange[Template:] _:blue[{prompt1_template}]_" )
    person = st.text_input("person:",value="einstein")
    st.markdown(f":orange[Template:] _:blue[{prompt2_template}]_" )
    language = st.text_input("language:",value="spanish")
    submit = st.form_submit_button("Submit",type="primary")

    if submit:
        with st.spinner(f"Running..."):
            chain1 = prompt1 | model | StrOutputParser()
            chain2 = (
                {"city": chain1, "language": itemgetter("language")}
                | prompt2
                | model
                | StrOutputParser()
            )

            response = chain2.invoke({"person": person , "language": language})
            st.write("Answer:")
            st.info(response)



