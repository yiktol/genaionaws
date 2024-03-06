import streamlit as st

# Delete all the items in Session state
for key in st.session_state.keys():
    del st.session_state[key]

st.set_page_config(
    page_title="Claude 3 Sonnet",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Claude 3 Sonnet")
st.markdown("""Claude 3 Sonnet by Anthropic strikes the ideal balance between intelligence and speed—particularly for enterprise workloads. \
It offers maximum utility at a lower price than competitors, and is engineered to be the dependable, high-endurance workhorse for scaled AI deployments. Claude 3 Sonnet can process images and return text outputs, and features a 200K context window.
            """)