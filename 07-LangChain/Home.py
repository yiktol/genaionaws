import streamlit as st

st.session_state.messages = []

st.set_page_config(
    page_title="LangChain",
    page_icon=":chains:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LangChain")
st.markdown("""LangChain is a framework for developing applications powered by language models. It enables applications that:\n
- Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
- Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)         
\nThis framework consists of several parts.

- LangChain Libraries: The Python and JavaScript libraries. Contains interfaces and integrations for a myriad of components, a basic run time for combining these components into chains and agents, and off-the-shelf implementations of chains and agents.
- LangChain Templates: A collection of easily deployable reference architectures for a wide variety of tasks.
- LangServe: A library for deploying LangChain chains as a REST API.
- LangSmith: A developer platform that lets you debug, test, evaluate, and monitor chains built on any LLM framework and seamlessly integrates with LangChain.
         
         """)

st.image("langchain_stack.svg")