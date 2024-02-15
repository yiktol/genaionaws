import streamlit as st

st.session_state.messages = []

st.set_page_config(
    page_title="Use Cases",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Use Cases")