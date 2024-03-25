import streamlit as st



st.set_page_config( 
    page_title="Challenge",  
    page_icon=":rock:",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key in st.session_state.keys():
    del st.session_state[key]

st.title("Prompt Engineering Exercise")


st.markdown("_Practice isn't the thing you do once you're good. It's the thing you do that makes you good._ - Malcolm Gladwell")
st.write("---")
st.markdown("""#### Please attempt the following Exercises.\n
Do bear in mind that there is no single one way to arrive at the same AI responses.\n
Also, the performance of the AI may change over time, and the responses you see here may not always be the same as what you'll get.
            """)