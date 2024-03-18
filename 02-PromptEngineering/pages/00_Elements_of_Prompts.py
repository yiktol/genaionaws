import streamlit as st
from langchain_community.llms import Bedrock
import utils.helpers as helpers

helpers.set_page_config()
bedrock_runtime = helpers.bedrock_runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)


prompt = """Summarize the following restaurant review

Restaurant: Luigi's, Location: Naples, Italy, Specialty: Pasta

Review: We were passing through SF on a Thursday afternoon and wanted some Italian food. We passed by a couple places which were packed until finally stopping at Luigi's, mainly because it was a little less crowded and the people seemed to be mostly locals. We ordered the tagliatelle and mozzarella caprese. The tagliatelle were a work of art - the pasta was just right and the tomato sauce with fresh basil was perfect. The caprese was OK but nothing out of the ordinary. Service was slow at first but overall it was fine. Other than that - Luigi's great experience!

Summary:"""

t1 = '''
For a summarization task, the prompt is a passage of text, and the model must respond with a shorter passage that captures the main points of the input. Specification of the output in terms of length (number of sentences or paragraphs) is helpful for this use case.
'''

if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "height" not in st.session_state:
    st.session_state.height = 200
if "provider" not in st.session_state:
    st.session_state.provider = "Amazon"
if "desc" not in st.session_state:
    st.session_state.desc = t1
if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = prompt


with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Elements of Prompts")
        st.markdown(st.session_state.desc)
    with col2:
        with st.container(border=True):
            provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
            model_id=st.text_input('model_id',helpers.getmodelId(provider))




row1_col1, row1_col2 = st.columns([0.5,0.5])





def call_llm(prompt):
    llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=helpers.getmodelparams(provider))
    response = llm.invoke(prompt)
    # Print results
    return st.info(response)


with st.container():
    
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.image("images/elements_of_prompts.png",width=500)
    with col2:
        with st.form("myform"):
            text_prompt = st.text_area(":orange[User Prompt:]", 
                                    height = 400,
                                    disabled = False,
                                    value = prompt)
            submitted = st.form_submit_button("Submit",type='primary')
        if text_prompt and submitted:
            st.write("Answer")
            with st.spinner("Thinking..."):
                call_llm(text_prompt)

