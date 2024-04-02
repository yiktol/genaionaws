import streamlit as st
import json
from langchain_community.llms import Bedrock
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.chains import LLMMathChain

import utils.helpers as helpers
bedrock = helpers.runtime_client()
helpers.set_page_config()


if "input" not in st.session_state:
    st.session_state.input = ""


st.header("ReAct Prompting")
st.markdown("""ReAct is a general paradigm that combines reasoning and acting with LLMs. \
ReAct prompts LLMs to generate verbal reasoning traces and actions for a task.
            """)


openai_api_key=json.loads(helpers.get_secret("SerpApiKey"))["OPENAI_API_KEY"]

llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)
# llm = Bedrock(
#     model_id='amazon.titan-tg1-large',
#     client=bedrock,
#     model_kwargs=helpers.getmodelparams('Amazon'),
# )


llm_math = LLMMathChain.from_llm(llm=llm)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)
tools = [math_tool,wiki]


options = [
    {"prompt_type":"Using Calulator Tool","prompt": "What is (4.5*2.1)^2.2?"},
    {"prompt_type":"Using Wikipedia Tool","prompt": "Who is Rosalind Franklin?"}
    ]

def update_options(item_num):
    st.session_state.input = options[item_num]["prompt"]
def load_options(item_num):
    st.write(f'Prompt: {options[item_num]["prompt"]}')
    st.button(f'Load Prompt', key=item_num, on_click=update_options, args=(item_num,))


col1, col2 = st.columns(2)

with col1:
    st.image("images/react.png")
    st.write("Calculator Example")
    st.image("images/calculator.png")

with col2:
   
    with st.container(border=True):
        st.write(":orange[Prompt Examples]")
        tab1, tab2 = st.tabs(["Calculator", "Wikipedia"])
        with tab1:
            load_options(item_num=0)
        with tab2:
            load_options(item_num=1)

    with st.form("form1"):
        input = st.text_area("Prompt:", key="input")
        submit = st.form_submit_button("Submit",type="primary")

    if submit:
        with st.spinner(f"Thinking..."):
            prompt = hub.pull("hwchase17/openai-functions-agent")
            llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)
            agent = create_openai_functions_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
            response = agent_executor.invoke({"input": input})
            st.write("ReAct:")
            st.write(response)
            st.write("Answer:")
            st.info(response["output"])
            



