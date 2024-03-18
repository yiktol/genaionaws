from langchain_community.llms import Bedrock
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

row1_col1, row1_col2 = st.columns([0.7,0.3])


t = '''
### Provide simple, clear, and complete instructions

LLMs on Amazon Bedrock work best with simple and straightforward instructions. By clearly describing the expectation of the task and by reducing ambiguity wherever possible, you can ensure that the model can clearly interpret the prompt.

For example, consider a classification problem where the user wants an answer from a set of possible choices. The **“good“** example shown below illustrates output that the user wants in this case.

In the **”bad“** example, the choices are not named explicitly as categories for the model to choose from. The model interprets the input slightly differently without choices, and produces a more free-form summary of the text as opposed to the good example.
'''

good_prompt = """The most common cause of color blindness is an inherited problem or variation in the functionality \
of one or more of the three classes of cone cells in the retina, which mediate color vision.\n
What is the above text about?
a) biology
b) history
c) geology """

bad_prompt = """Classify the following text.\n
\"The most common cause of color blindness is an inherited problem or variation in the functionality \
of one or more of the three classes of cone cells in the retina, which mediate color vision.\""""

if "prompt" not in st.session_state:
    st.session_state.prompt = bad_prompt
if "height" not in st.session_state:
    st.session_state.height = 120
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Bad Prompt"

options = [{"prompt_type":"Good Prompt", "prompt": good_prompt, "height":200},
            {"prompt_type":"Bad Prompt", "prompt": bad_prompt, "height":120}]

def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
def load_options(item_num):
    st.write(f'Prompt: {options[item_num]["prompt"]}')
    st.button(f'Load Prompt', key=item_num, on_click=update_options, args=(item_num,))

with st.container():
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Good vs Bad Prompt")
        st.markdown(t)
        with st.form("myform2"):
            prompt_data = st.text_area(f":orange[{st.session_state.prompt_type}:]", height = int(st.session_state.height),key="prompt" )
            submit = st.form_submit_button("Submit", type="primary")
    with col2:
        with st.container(border=True):
            provider = st.selectbox('Provider',['Amazon','Anthropic','AI21','Cohere','Meta'])
            model_id=st.text_input('model_id',helpers.getmodelId(provider))
        with st.container(border=True):
            tab1, tab2 = st.tabs(["Good Prompt", "Bad Prompt"])
            with tab1:
                load_options(item_num=0)
            with tab2:
                load_options(item_num=1)

#Create the connection to Bedrock
bedrock_runtime = helpers.bedrock_runtime_client()


llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=helpers.getmodelparams(provider),
)


if submit:
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt_data)

            #print(response)
            st.write("### Answer")
            st.info(response)
  