import streamlit as st
from langchain_community.llms import Bedrock
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from helpers import bedrock_runtime_client, set_page_config

bedrock = bedrock_runtime_client()

llm = Bedrock(
    client=bedrock, model_id="amazon.titan-text-express-v1"
)

set_page_config()


st.header("Output Parsers")
st.markdown("""Output parsers are responsible for taking the output of an LLM and transforming it to a more suitable format. \
This is very useful when you are using LLMs to generate any form of structured data.
            """)

st.subheader(":orange[Pydantic parser]")
st.markdown("This output parser allows users to specify an arbitrary Pydantic Model and query LLMs for outputs that conform to that schema.")

expander = st.expander("See code")
expander.code("""from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_community.llms import Bedrock

model = Bedrock(
    client=bedrock, model_id="amazon.titan-text-express-v1"
)

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\\n{format_instructions}\\n{query}\\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | model
output = prompt_and_model.invoke({"query": "Tell me a joke."})
parser.invoke(output)""",language="python")



# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup",allow_reuse=True)
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
template="Answer the user query.\n{format_instructions}\n{query}\n",
input_variables=["query"],
partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt_and_model = prompt | llm
#print(prompt.format(query="Tell me a joke."))
# st.write("Prompt Template:")
# st.write(prompt.format(query="Tell me a joke."))
# And a query intended to prompt a language model to populate the data structure.

with st.form("form1"):
    prompt_data = st.text_area("Prompt:", value="Tell me a joke.")
    submit = st.form_submit_button("Submit",type="primary")

    if submit:
        #print(response)
        st.write("Answer:")
        with st.spinner("AI Thinking..."):
            output = prompt_and_model.invoke({"query": prompt_data[0]})
            parse_data = parser.invoke(output)
            st.write(output)
            st.write("Parsed Data:")
            st.write(f"Joke({parse_data})")


st.subheader(":orange[CSV parser]")
st.markdown("This output parser can be used when you want to return a list of comma-separated items.")


expander = st.expander("See code")
expander.code("""from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

model = Bedrock(
    client=bedrock, model_id="amazon.titan-text-express-v1"
)

chain = prompt | model | output_parser
chain.invoke({"subject": "ice cream flavors"})""",language="python")



output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | output_parser

with st.form("form2"):
    prompt_data = st.text_area("Prompt:", value="ice cream flavors")
    submit2 = st.form_submit_button("Submit",type="primary")

    if submit2:
        #print(response)
        st.write("Answer:")
        with st.spinner("AI Thinking..."):
            output = chain.invoke({"subject": prompt_data})
            st.write(output)

