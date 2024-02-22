from typing import Any, Dict, List, Union
import streamlit as st
import json
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain.chains import LLMMathChain
from langchain.agents import Tool

from helpers import bedrock_runtime_client, set_page_config, get_secret
bedrock = bedrock_runtime_client()
set_page_config()

openai_api_key=json.loads(get_secret("SerpApiKey"))["OPENAI_API_KEY"]


st.header("Callbacks")
st.markdown("""LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. \
This is useful for logging, monitoring, streaming, and other tasks.""")

st.subheader("Multiple callback handlers")
st.markdown("""In many cases, it is advantageous to pass in handlers instead when running the object. \
When we pass through CallbackHandlers using the callbacks keyword arg when executing an run, \
those callbacks will be issued by all nested objects involved in the execution. \
For example, when a handler is passed through to an Agent, it will be used for all callbacks related to the agent and all the objects involved in the agent's execution, in this case, the Tools, LLMChain, and LLM.
""")

expander = st.expander("See code")
expander.code("""from typing import Any, Dict, List, Union

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain_openai import OpenAI


# First, define custom callback handler implementations
class MyCustomHandlerOne(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"on_llm_start {serialized['name']}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        print(f"on_new_token {token}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        \"""Run when LLM errors.\"""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        print(f"on_chain_start {serialized['name']}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        print(f"on_tool_start {serialized['name']}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"on_agent_action {action}")


class MyCustomHandlerTwo(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"on_llm_start (I'm the second handler!!) {serialized['name']}")


# Instantiate the handlers
handler1 = MyCustomHandlerOne()
handler2 = MyCustomHandlerTwo()

# Setup the agent. Only the `llm` will issue callbacks for handler2
llm = OpenAI(temperature=0, streaming=True, callbacks=[handler2])
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Callbacks for handler1 will be issued by every object involved in the
# Agent execution (llm, llmchain, tool, agent executor)
agent.run("What is 2 raised to the 0.235 power?", callbacks=[handler1])
""",language="python")




# First, define custom callback handler implementations
class MyCustomHandlerOne(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        # print(f"on_llm_start {serialized['name']}")
        st.write(f"on_llm_start {serialized['name']}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        # print(f"on_new_token {token}")
        st.write(f"on_new_token {token}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        # print(f"on_chain_start {serialized['name']}")
        st.write(f"on_chain_start {serialized['name']}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        # print(f"on_tool_start {serialized['name']}")
        st.write(f"on_tool_start {serialized['name']}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        # print(f"on_agent_action {action}")
        st.write(f"on_agent_action {action}")


class MyCustomHandlerTwo(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        # print(f"on_llm_start (I'm the second handler!!) {serialized['name']}")
        st.write(f"on_llm_start (I'm the second handler!!) {serialized['name']}")


# Instantiate the handlers
handler1 = MyCustomHandlerOne()
handler2 = MyCustomHandlerTwo()

# Setup the agent. Only the `llm` will issue callbacks for handler2
llm = OpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key, callbacks=[handler2])
llm_math = LLMMathChain.from_llm(llm=llm)
math = Tool(
    name="llm_math",
    description="Math Function.",
    func=llm_math.run,
)


# tools = [math]
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)


# Callbacks for handler1 will be issued by every object involved in the
# Agent execution (llm, llmchain, tool, agent executor)
with st.form("my_form"):
    input = st.text_input("Prompt:", "What is 2 raised to the 0.235 power?")
    submitted = st.form_submit_button("Submit", type='primary')
    
    if submitted:
        st.write(":orange[Callbacks:]")
        # prompt = hub.pull("hwchase17/openai-functions-agent")
        # agent = create_openai_functions_agent(llm, tools,prompt)
        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
        response = agent.run(input, callbacks=[handler1])
        # response = agent_executor.invoke({"input": input},llm )
        st.write("Answer:")
        st.success(response)
        st.stop()
    else:
        st.stop()

