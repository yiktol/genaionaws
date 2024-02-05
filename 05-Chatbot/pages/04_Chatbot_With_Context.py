import streamlit as st
import boto3

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.llms.bedrock import Bedrock

from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from helpers import set_page_config

set_page_config()

st.title("Titan Chatbot with Context")

def form_callback():
    st.session_state.messages = []

st.sidebar.button(label='Clear Messages', on_click=form_callback)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

retriever =''
uploaded_file =''

documents_aws =''

bedrock_runtime = boto3.client(
service_name='bedrock-runtime',
region_name='us-east-1', 
)

br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)


modelId = "amazon.titan-tg1-large"
titan_llm = Bedrock(model_id=modelId, client=bedrock_runtime)
titan_llm.model_kwargs = {'temperature': 0.1, "maxTokenCount": 700}

@st.cache_data()
def load_csv(data):
    loader = CSVLoader(data) # --- > 219 docs with 400 chars
    documents_aws = loader.load() #
    return documents_aws

@st.cache_data()
def chunk_uploaded_file():
    documents_aws = load_csv(uploaded_file.name)
    docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separator=",").split_documents(documents_aws)
    print(f"Documents:after split and chunking size={len(docs)}")
    return docs

def vectorize():
    vectorstore_faiss_aws = FAISS.from_documents(
        documents=chunk_uploaded_file(),
        embedding = br_embeddings, 
        #**k_args
    )
    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_aws)
    retriever = vectorstore_faiss_aws.as_retriever()
    return (retriever, wrapper_store_faiss)


def qa_run(prompt,retriever):
    memory_chain = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
    chat_history=[]
    qa = ConversationalRetrievalChain.from_llm(
        llm=titan_llm, 
        retriever=retriever, 
        #retriever=vectorstore_faiss_aws.as_retriever(search_type='similarity', search_kwargs={"k": 8}),
        memory=memory_chain,
        verbose=True,
        #condense_question_prompt=CONDENSE_QUESTION_PROMPT, # create_prompt_template(), 
        chain_type='stuff', # 'refine',
        #max_tokens_limit=100
    )
    
    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
{context}

Use at maximum 3 sentences to answer the question inside the <q></q> XML tags. 

<q>{question}</q>

Do not use any XML tags in the answer. If the answer is not in the context say "Sorry, I don't know, as the answer was not found in the context."

Answer:""")
    return qa.run({'question': prompt })

def create_prompt_template():
    _template = """{chat_history}

Answer only with the new question.
How would you ask the question considering the previous conversation: {question}
Question:"""
    CONVO_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONVO_QUESTION_PROMPT
 

uploaded_file = st.file_uploader("Choose a CSV file")

    
button = st.button("Vectorize", type="primary")

if button:
    with st.spinner("Vectorizing..."):
        retriever = vectorize()[0]
        st.session_state.retriever = retriever 
        st.success("Done")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "User", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = qa_run(prompt,st.session_state.retriever)
            st.markdown(stream)
        #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})













