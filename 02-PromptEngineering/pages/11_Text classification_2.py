import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
   
row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Text classification")

t = '''
### Sentiment analysis

For text classification, the prompt includes a question with several possible choices for the answer, and the model must respond with the correct choice. Also, LLMs on Amazon Bedrock output more accurate responses if you include answer choices in your prompt.

Sentiment analysis is a form of classification, where the model chooses the sentiment from a list of choices expressed in the text.
'''

row1_col1.markdown(t)
with row1_col2.form(key ='Form1'):
        provider = st.selectbox('Provider',('Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

st.write(":orange[Template:]")
template = '''
The following is text from a {text_type}:\n
{context}\n
Tell me the sentiment of the {text_type} and categorize it as one of the following:\n
{sentimentA}
{sentimentB}
{sentimentC}

'''
st.code(template, language='text')


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
    
    prompt = PromptTemplate(input_variables=["context","text_type","sentimentA","sentimentB","sentimentC"], template=template)
    prompt_query = prompt.format(
            text_type="The following is text from a restaurant review:",
            context="I finally got to check out Alessandro’s Brilliant Pizza and it is now one of my favorite restaurants in Seattle. The dining room has a beautiful view over the Puget Sound but it was surprisingly not crowed. I ordered the fried castelvetrano olives, a spicy Neapolitan-style pizza and a gnocchi dish. The olives were absolutely decadent, and the pizza came with a smoked mozzarella, which was delicious. The gnocchi was fresh and wonderful. The waitstaff were attentive, and overall the experience was lovely. I hope to return soon.",
            sentimentA="Positive",
            sentimentB="Negative",
            sentimentC="Neutral"
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_area(":orange[User Prompt:]", 
                              height = 300,
                              disabled = False,
                              value = "The following is text from a restaurant review:\n\n“I finally got to check out Alessandro’s Brilliant Pizza and it is now one of my favorite restaurants in Seattle. The dining room has a beautiful view over the Puget Sound but it was surprisingly not crowed. I ordered the fried castelvetrano olives, a spicy Neapolitan-style pizza and a gnocchi dish. The olives were absolutely decadent, and the pizza came with a smoked mozzarella, which was delicious. The gnocchi was fresh and wonderful. The waitstaff were attentive, and overall the experience was lovely. I hope to return soon.”\n\nTell me the sentiment of the restaurant review and categorize it as one of the following:\n\nPositive\nNegative\nNeutral")
    submitted = st.form_submit_button("Submit")
if topic_text and submitted:
    call_llm()
