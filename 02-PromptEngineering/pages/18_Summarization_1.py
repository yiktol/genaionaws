import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
    
    
st.title("Summarization")
t = '''
### Summarization

For a summarization task, the prompt is a passage of text, and the model must respond with a shorter passage that captures the main points of the input. Specification of the output in terms of length (number of sentences or paragraphs) is helpful for this use case.
'''

st.markdown(t)
st.write("**:orange[Template:]**")
template = '''
The following is text from a {Category}:\n
\"{Context}\"\n
Summarize the above {Category} in {length_of_summary}
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(category,context,length_of_summary):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Category","Context","length_of_summary"], template=template)
    prompt_query = prompt.format(
            Category=category,
            Context=context,
            length_of_summary=length_of_summary)
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    category = "restaurant review"
    context = '''I finally got to check out Alessandro\'s Brilliant Pizza \
and it is now one of my favorite restaurants in Seattle. \
The dining room has a beautiful view over the Puget Sound \
but it was surprisingly not crowed. I ordered the fried \
castelvetrano olives, a spicy Neapolitan-style pizza \
and a gnocchi dish. The olives were absolutely decadent, \
and the pizza came with a smoked mozzarella, which was delicious. \
The gnocchi was fresh and wonderful. The waitstaff were attentive, \
and overall the experience was lovely. I hope to return soon.'''
    length_of_summary = "one sentence"
    text_prompt = st.text_area(":orange[User Prompt:]", 
                              height = 200,
                              disabled = False,
                              value = (f"The following is text from a {category}:\n\n\"{context}\"\n\nSummarize the above {category} in {length_of_summary}."))
    submitted = st.form_submit_button("Submit")
if text_prompt and submitted:
    call_llm(category,context,length_of_summary)
