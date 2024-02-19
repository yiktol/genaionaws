import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from helpers import bedrock_runtime_client, set_page_config


set_page_config()
bedrock = bedrock_runtime_client()

embeddings = BedrockEmbeddings(
    client=bedrock, region_name="us-east-1"
)


st.header("Vector stores")
st.markdown("""One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. \
A vector store takes care of storing embedded data and performing vector search for you.
            """)

st.image("vector_stores.jpg")

expander = st.expander("See code")
expander.code("""from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

embeddings = BedrockEmbeddings(
    client=bedrock, region_name="us-east-1"
)

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('2022-letter.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, embeddings)

query = "How Amazon perform in 2022?"
docs = db.similarity_search(query)
print(docs[0].page_content)
    """,language="python")

with st.form("form1"):
    uploaded_file = st.text_input("Letter", "2022-letter.txt",disabled=True)
    query = st.text_input("Query", "How Amazon perform in 2022?",disabled=False)
    submit = st.form_submit_button("Load Data",type="primary")

if submit:
    with st.spinner("Loading TXT...."):
        raw_documents = TextLoader('2022-letter.txt').load()
        st.success("Done!, Loading")
    with st.spinner("Splitting TXT...."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)
        st.info(f"You have {len(documents)} documents")
        st.success("Done!, Splitting")
    with st.spinner("Embedding TXT...."):   
        db = FAISS.from_documents(documents, embeddings)
        embedding_list = embeddings.embed_documents([document.page_content for document in documents])

        st.info(f"You have {len(embedding_list)} embeddings")
        st.info(f"Here's a sample of one: {embedding_list[0][:10]}...")
        st.success("Done!, Embedding")
        
    with st.spinner("Similarity search...."):
        docs = db.similarity_search(query)
        st.write(f"Here are the most similar documents to \"{query}\":")
        st.info(docs[0].page_content)
        st.success("Done!, Similarity search")
