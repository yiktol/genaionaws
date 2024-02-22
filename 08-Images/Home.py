import streamlit as st

st.session_state.messages = []

st.set_page_config(
    page_title="Images",
    page_icon=":chains:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Images")
