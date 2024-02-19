import streamlit as st

st.session_state.messages = []

st.set_page_config(
    page_title="LangChain",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LangChain")
st.markdown("""LangChain is a framework for developing applications powered by language models. It enables applications that:\n
- Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
- Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)         
         
         """)