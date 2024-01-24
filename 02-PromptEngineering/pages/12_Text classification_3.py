import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Text classification")

t = '''
### Generate output enclosed in XML tags

The following example uses Claude models to classify text. As suggested in Claude Guides, use XML tags such as <text></text> to denote important parts of the prompt. Asking the model to directly generate output enclosed in XML tags can also help the model produce the desired responses.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Anthropic','AI21'),disabled=True)
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

st.write(":orange[Template:]")
template = '''
Human: {task}\n
<text>{context}</text>\n
Categories are:\n
{category1}
{category2}
{category3}\n
Assistant:
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm():
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["context","taske","category1","category2","category3"], template=template)
    prompt_query = prompt.format(
            task="Classify the given product description into given categories. Please output the category label in <output></output> tags.\n\nHere is the product description.",
            context="Safe, made from child-friendly materials with smooth edges. Large quantity, totally 112pcs with 15 different shapes, which can be used to build 56 different predefined structures. Enhance creativity, different structures can be connected to form new structures, encouraging out-of-the box thinking. Enhance child-parent bonding, parents can play with their children together to foster social skills.",
            category1="(1) Toys",
            category2="(2) Beauty and Health",
            category3="(3) Electronics"
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_area(":orange[User Prompt:]", 
                              height = 380,
                              disabled = False,
                              value = "Human: Classify the given product description into given categories. Please output the category label in <output></output> tags.\n\nHere is the product description.\n\n<text>\nSafe, made from child-friendly materials with smooth edges. Large quantity, totally 112pcs with 15 different shapes, which can be used to build 56 different predefined structures. Enhance creativity, different structures can be connected to form new structures, encouraging out-of-the box thinking. Enhance child-parent bonding, parents can play with their children together to foster social skills.\n</text>\n\nCategories are:\n(1) Toys\n(2) Beauty and Health\n(3) Electronics\n\nAssistant: ")
    submitted = st.form_submit_button("Submit")
if topic_text and submitted:
    call_llm()
