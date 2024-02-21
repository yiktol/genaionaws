import streamlit as st
import json
from langchain_community.utilities import SerpAPIWrapper
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import Tool

from helpers import bedrock_runtime_client, set_page_config, get_secret
bedrock = bedrock_runtime_client()
set_page_config()




st.header("Agents")
st.markdown("""The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.
            """)

st.subheader("Tools")
st.markdown("""Tools are the core concept of agents. Tools are used to perform actions. \
Tools are defined by a name, a description, and a function. Tools are used to perform actions. \
Tools are defined by a name, a description, and a function.
- **Wikipedia** is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. Wikipedia is the largest and most-read reference work in history.
- **SerpApi** is a real-time API to access Google search results. We handle proxies, solve captchas, and parse all rich structured data for you.
""")

expander = st.expander("See code")
expander.code("""from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_openai import ChatOpenAI

serpapi_api_key=json.loads(get_secret("Keys"))["SERP_API_KEY"]
openai_api_key=json.loads(get_secret("Keys"))["OPENAI_API_KEY"]

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
google = Tool(
    name="serpapi",
    description="Google Search.",
    func=search.run,
)

tools = [wiki,google]

prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

response = agent_executor.invoke({"input": "What are the famous books of Jose Rizal?"})

""",language="python")

serpapi_api_key=json.loads(get_secret("SerpApiKey"))["SERP_API_KEY"]
openai_api_key=json.loads(get_secret("SerpApiKey"))["OPENAI_API_KEY"]

search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
google = Tool(
    name="serpapi",
    description="Google Search.",
    func=search.run,
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wiki,google]

# Get the prompt to use - you can modify this!
# If you want to see the prompt in full, you can at: https://smith.langchain.com/hub/hwchase17/openai-functions-agent


with st.form("form1"):
    input = st.text_area("Prompt:", value="What are the famous books of Jose Rizal?" )
    submit = st.form_submit_button("Submit",type="primary")

    if submit:
        with st.spinner(f"Agent Running...Invoking {wiki.name}"):
            prompt = hub.pull("hwchase17/openai-functions-agent")
            llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)
            agent = create_openai_functions_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
            response = agent_executor.invoke({"input": input})
            st.write("Answer:")
            st.write(response)