import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

with st.sidebar:
    "Parameters:"
    with st.form(key ='Form1'):
        provider = st.selectbox('Provider',('Antropic','Amazon','AI21'))
        model_id=st.text_input('model_id',getmodelId(provider))
        submitted1 = st.form_submit_button(label = 'Set Parameters') 
    
    
st.title("Question-answer")
t = '''
### Question-answer, with context

When prompting Claude models, it's helpful to wrap the input text in XML tags. In the following example, the input text is enclosed in <text></text>.
'''

st.markdown(t)
st.write("**:orange[Template:]**")
template = '''
Human: {Instruction}\n
<text>\n
{Context}\n
</text>\n
{Question}\n

Assistant:
'''
st.code(template, language='None')

#Create the connection to Bedrock
bedrock_runtime = bedrock_runtime_client()

def call_llm(question,context,instruction):
    # Instantiate LLM model
    llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=getmodelparams(provider)
    )
    # Prompt
    
    prompt = PromptTemplate(input_variables=["Question","Context","Instruction"], template=template)
    prompt_query = prompt.format(
            Question=question,
            Context=context,
            Instruction=instruction
            )
    print(prompt_query)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    instruction = "Read the following text inside <text></text> XML tags, and then answer the question:"
    context = ("On November 12, 2020, the selection of the Weekend to headline the show was announced; marking the first time a Canadian solo artist headlined the Super Bowl halftime show. When asked about preparations for the show, the Weekend stated, \"We've been really focusing on dialing in on the fans at home and making performances a cinematic experience, and we want to do that with the Super Bowl.\"\n\nThe performance featured a choir whose members were dressed in white and wore masks over their faces with glowing red eyes, and were standing within a backdrop of a neon cityscape. The performance opened with a white figure dressed the same as the choir being lowered into the backdrop where the choir was standing while singing “Call Out My Name\". At this time, the Weekend sat in a convertible against a skyline backdrop designed to resemble the Las Vegas Strip. For the next part of the performance, the backdrop then split open to reveal the Weekend, who then performed \"Starboy\", followed by \"The Hills\".\n\nNext, performing the song \"Can't Feel My Face\", the Weekend traveled through a labyrinth constructed behind the stage, joined by dancers dressed in red blazers and black neckties similar to his, but with their faces covered with bandages, in keeping with the aesthetic of his fourth studio album After Hours (2020). The dancers would wear these bandages throughout the performance. In the labyrinth section of the performance, camerawork was visually unsteady. The next songs performed were \"I Feel It Coming\", \"Save Your Tears\", and \"Earned It\". For the \"Earned It\" performance, the Weekend was accompanied by violinists. For the finale of the show, the Weekend took to the field of the stadium with his dancers to perform “Blinding Lights\". He and the dancers entered the field by performing \"House of Balloons / Glass Table Girls\". The performance ended with an array of fireworks.")
    question = "Based on the text above, what songs did the Weekend play at the Super Bowl halftime show?"
    text_prompt = st.text_area(":orange[User Prompt:]", 
                              height = 500,
                              disabled = False,
                              value = (f"Human: {instruction}\n\n<text>\n{context}\n</text>\n\n{question}\n\nAssistant:"))
    submitted = st.form_submit_button("Submit")
if text_prompt and submitted:
    call_llm(question,context,instruction)
