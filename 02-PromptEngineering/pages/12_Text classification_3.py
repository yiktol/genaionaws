import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        # provider = st.selectbox('Provider',('Antropic'))
        model_id=st.text_input('model_id',getmodelId('Antropic'))
        # temperature = st.number_input('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        # top_k=st.number_input('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        # top_p=st.number_input('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        # max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 4096, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
    
    
st.title("Text classification")
t = '''
### Generate output enclosed in XML tags

The following example uses Claude models to classify text. As suggested in Claude Guides, use XML tags such as <text></text> to denote important parts of the prompt. Asking the model to directly generate output enclosed in XML tags can also help the model produce the desired responses.
'''

st.markdown(t)
st.write("**:orange[Template:]**")
template = '''
Human: {task}\n
<text>{context}</text>\n
Categories are:\n
{category1}\n
{category2}\n
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
    model_kwargs=getmodelparams('Antropic')
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
