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


st.title("Claude Models")
st.markdown("""Claude models
Anthropic's Claude family of models - Haiku, Sonnet, and Opus - allow customers to choose the exact combination of intelligence, speed, and cost that suits their business needs. \
Claude 3 Opus, the company's most capable model, has set a market standard on benchmarks. \
All of the latest Claude models have vision capabilities that enable them to process and analyze image data, meeting a growing demand for multimodal AI systems that can handle diverse data formats. \
While the family offers impressive performance across the board, Claude 3 Haiku is one of the most affordable and fastest options on the market for its intelligence category.
            """)

st.subheader("Claude 3 Sonnet")
st.markdown("""Claude 3 Sonnet by Anthropic strikes the ideal balance between intelligence and speed—particularly for enterprise workloads. \
It offers maximum utility at a lower price than competitors, and is engineered to be the dependable, high-endurance workhorse for scaled AI deployments. Claude 3 Sonnet can process images and return text outputs, and features a 200K context window.
            """)