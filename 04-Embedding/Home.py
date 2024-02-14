import streamlit as st

st.session_state.messages = []

st.set_page_config(
    page_title="Embedding",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Embedding")
st.sidebar.success("Select a page above.")

st.markdown("""The academic definition of embedding is translating high-dimensional vectors into a relatively low-dimensional space. \
You might know each and every word in this sentence but still have no idea about the whole sentence. \
We can think of embedding as converting natural language into a sequence of numbers, \
with the input being a piece of text and the output being a vector. \
In other words, the vector is a numerical representation of the text, \
making it easy to perform all kinds of complex computations in AI/ML.""")

st.image('embedding.png')

st.page_link("https://www.pinecone.io/learn/vector-embeddings/", label="What are Vector Embeddings", icon="🌎")