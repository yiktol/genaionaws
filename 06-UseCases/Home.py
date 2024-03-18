import streamlit as st

st.session_state.messages = []

st.set_page_config(
    page_title="Use Cases",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Use Cases")
st.markdown("Where is Generative AI having the most impact?")
st.image("images/use_cases1.png", width=800)
st.markdown("""Here are a couple of example use cases that are powered by foundation models. 
- Foundation models can be used to create  novel protein sequences with specific properties for antibody design ,enzyme design, gene therapy and vaccine design
- Foundation models can generate synthetic data to augment datasets, to improve the performance of AI models to address privacy concerns
- One of the niches in which generative AI will have the most substantial impact is customer service and call centers. Language models are revolutionizing customer service conversations as they automate pre-call, in-call, and post-call activities like after-call documentation, agent coaching, and summarization. When combined with advanced text-to-speech technology, these language models will soon take over the entire customer engagement process without needing human interference; unlike traditional call centers that follow strict rules, this transitions into an easygoing, natural conversation indistinguishable from an actual human. Almost all conversations your business has with consumers on any subject will be automated
- Conventional chatbots are usually scripted and lack sufficient machine learning and natural language processing capabilities. Nowadays, however, AI-powered chatbots that leverage external databases have become adept at responding swiftly to complex customer queries, holding more meaningful discussions, and escalating the conversation to humans when necessary.
""")

st.image("images/use_cases2.png")
