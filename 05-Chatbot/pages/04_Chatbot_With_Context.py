import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import BedrockChat
from langchain.schema import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


from helpers import set_page_config, bedrock_runtime_client

set_page_config()

st.title("Chatbot with Context")
st.write("""In this use case we will ask the Chatbot to answer question from the context that it was passed. \
We will take a csv file and use Titan embeddings Model to create the vector. \
This vector is stored in FAISS. When chatbot is asked a question we pass this vector and retrieve the answer.""")

def form_callback():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.retriever = None
    st.session_state.retrieval_chain = None

st.sidebar.button(label='Clear Chat Messages', on_click=form_callback)

# Initialize
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


documents_aws =''

bedrock = bedrock_runtime_client()
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


modelId = "anthropic.claude-v2"
llm = BedrockChat(model_id=modelId, client=bedrock)
llm.model_kwargs = {'temperature': 0.1}

@st.cache_data()
def load_csv(data):
    loader = CSVLoader(data) # --- > 219 docs with 400 chars
    documents_aws = loader.load() #
    return documents_aws

@st.cache_data()
def chunk_uploaded_file(uploaded_file):
    documents_aws = load_csv(uploaded_file)
    docs = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator=",").split_documents(documents_aws)
    print(f"Documents:after split and chunking size={len(docs)}")
    return docs

def vectorize(uploaded_file):
    vectorstore_faiss_aws = FAISS.from_documents(
        documents=chunk_uploaded_file(uploaded_file),
        embedding = embeddings
    )
    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_aws)
    retriever = vectorstore_faiss_aws.as_retriever()
    return retriever

retrieval_qa_chat_prompt  = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI and you answer question only if it is in the context. If the question is not in the context, \
        you will politely explain that the statement is not within the context.:\n\n{context}")]
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)


uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file:
    filename = uploaded_file.name
button = st.button("Process File", type="primary")

if button:
    with st.spinner("Vectorizing..."):
        retriever = vectorize(filename)
        #print(type(retriever))
        st.session_state.retriever = retriever
        
        retrieval_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)
        st.session_state.retrieval_chain = retrieval_chain
        st.success("Done")
               
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Say something"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "User", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = st.session_state.retrieval_chain.invoke({"input": prompt})
            st.write(stream["answer"])
        #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream["answer"]})













