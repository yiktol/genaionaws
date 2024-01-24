import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Summarization")

t = '''
### Summarization

In the following example, Claude summarizes the given text in one sentence. To include input text in your prompts, format the text with XML mark up: <text> {text content} </text>. Using XML within prompts is a common practice when prompting Claude models.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Anthropic','Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
    

st.write(":orange[Template:]")
template = '''Human: Please read the text:\n
<text>
{Context}
</text>\n
Summarize the text in {{length of summary, e.g., “one sentence” or “one paragraph”}}\n
Assistant:
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(context,length_of_summary):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    prompt = PromptTemplate(input_variables=["Context","length_of_summary"], template=template)
    prompt_query = prompt.format(
            Context=context,
            length_of_summary=length_of_summary
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    context = '''In game theory, the Nash equilibrium, named after the mathematician John Nash, is the most common way to define the solution of a non-cooperative game involving two or more players. \
In a Nash equilibrium, each player is assumed to know the equilibrium strategies of the other players, and no one has anything to gain by changing only one's own strategy. \
The principle of Nash equilibrium dates back to the time of Cournot, who in 1838 applied it to competing firms choosing outputs.'''
    length_of_summary = "one sentence"
    text_prompt = st.text_area(":orange[User Prompt:]", 
                              height = 300,
                              disabled = False,
                              value = (f"Human: Please read the text:\n\n<text>\n{context}\n</text>\n\nSummarize the text in {length_of_summary}.\n\nAssistant:"))
    submitted = st.form_submit_button("Submit")
if text_prompt and submitted:
    call_llm(context,length_of_summary)
