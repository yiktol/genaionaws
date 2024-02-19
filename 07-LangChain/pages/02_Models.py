import streamlit as st
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import Bedrock
from helpers import bedrock_runtime_client, set_page_config

set_page_config()
bedrock = bedrock_runtime_client()

st.header("Models")
st.markdown("Large Language Models (LLMs) are a core component of LangChain. \
LangChain does not serve its own LLMs, but rather provides a standard interface for interacting with many different LLMs. \
To be specific, this interface is one that takes as input a string and returns a string.")

st.subheader(":orange[Language Model]")
st.markdown("Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. ")

expander = st.expander("See code")
expander.code(f"""from langchain_community.llms import Bedrock

llm = Bedrock(
    credentials_profile_name="bedrock-admin", model_id="amazon.titan-text-express-v1"
)

llm.invoke("What day comes after Friday?")
        """,language="python")

with st.form("form1"):
    prompt = st.text_input("Prompt:", value="What day comes after Friday?")
    submit = st.form_submit_button("Submit",type="primary")

    llm = Bedrock(
        client=bedrock, model_id="amazon.titan-text-express-v1"
    )

    if submit:
        output = llm.invoke(prompt)
        #print(output)
        st.info(output)


st.subheader(":orange[Chat Model]")
st.markdown("A model that takes a series of messages and returns a message output.")

expander2 = st.expander("See code")
expander2.code(f"""from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, SystemMessage

chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={{"temperature": 0.1}})

messages = [
    SystemMessage(content="You are a helpful AI bot that makes a joke at whatever the user says"),
    HumanMessage(
        content="Roses are Red"
    )
]
chat(messages)""",language="python")



from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = BedrockChat(client=bedrock,model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})


container = st.container(border=True)
container.write(":orange[Chat:]")
prompt = container.chat_input("Say something")
if prompt:
    container.write(f"Human: {prompt}")
    
    messages = [
        SystemMessage(content="You are a helpful AI bot that makes a joke at whatever the user says"),
        HumanMessage(content=prompt)
    ]
    with container.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = chat.invoke(messages)
            st.write(response.content)


st.subheader(":orange[Text Embedding Model]")
st.markdown("Text embeddings model that converts natural language text including single words, phrases, or even large documents, into numerical representations that can be used to power use cases such as search, personalization, and clustering based on semantic similarity.")

expander3 = st.expander("See code")
expander3.code(f"""from langchain_community.embeddings import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    credentials_profile_name="bedrock-admin", region_name="us-east-1"
)

embeddings.embed_query("This is a content of the document")

""",language="python")


from langchain_community.embeddings import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    client=bedrock, region_name="us-east-1"
)


with st.form("form3"):
    prompt2 = st.text_input("Prompt:", value="What day comes after Friday?")
    submit2 = st.form_submit_button("Submit",type="primary")

    if submit2:
        embedding_value = embeddings.embed_query(prompt2)
        st.info(embedding_value[:10])
        st.info(f"Embedding Dimension: {len(embedding_value)}")