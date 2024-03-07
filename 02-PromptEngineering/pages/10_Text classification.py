import streamlit as st
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()
bedrock_runtime = bedrock_runtime_client()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)

if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "height" not in st.session_state:
    st.session_state.height = 200
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "Prompt"
if "provider" not in st.session_state:
    st.session_state.provider = "Amazon"
if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = ""
if "desc" not in st.session_state:
    st.session_state.desc = ""


row1_col1, row1_col2 = st.columns([0.7,0.3])

row1_col1.title("Text classification")

t1 = '''
### Multiple-choice classification question

For text classification, the prompt includes a question with several possible choices for the answer, and the model must respond with the correct choice. Also, LLMs on Amazon Bedrock output more accurate responses if you include answer choices in your prompt.

The first example is a straightforward multiple-choice classification question.
'''
t2 = '''
### Sentiment analysis

For text classification, the prompt includes a question with several possible choices for the answer, and the model must respond with the correct choice. Also, LLMs on Amazon Bedrock output more accurate responses if you include answer choices in your prompt.

Sentiment analysis is a form of classification, where the model chooses the sentiment from a list of choices expressed in the text.
'''
t3 = '''
### Generate output enclosed in XML tags

The following example uses Claude models to classify text. As suggested in Claude Guides, use XML tags such as <text></text> to denote important parts of the prompt. Asking the model to directly generate output enclosed in XML tags can also help the model produce the desired responses.
'''
template1 = '''
{context}\n
{question}? Choose from the following:\n
{choice1}
{choice2}
{choice3}
'''
template2 = '''
The following is text from a {text_type}:\n
{context}\n
Tell me the sentiment of the {text_type} and categorize it as one of the following:\n
{sentimentA}
{sentimentB}
{sentimentC}
'''
template3 = '''
Human: {task}\n
<text>{context}</text>\n
Categories are:\n
{category1}
{category2}
{category3}\n
Assistant:
'''
prompt1 = PromptTemplate(input_variables=["context","question","cloice1","choice2","choice3"], template=template1)
prompt_query1 = prompt1.format(
        context="San Francisco, officially the City and County of San Francisco, is the commercial, financial, and cultural center of Northern California. The city proper is the fourth most populous city in California, with 808,437 residents, and the 17th most populous city in the United States as of 2022.",
        question="What is the paragraph above about",
        choice1="A city",
        choice2="A person",
        choice3="An event"
        )
prompt2 = PromptTemplate(input_variables=["context","text_type","sentimentA","sentimentB","sentimentC"], template=template2)
prompt_query2 = prompt2.format(
        text_type="The following is text from a restaurant review:",
        context="I finally got to check out Alessandro's Brilliant Pizza and it is now one of my favorite restaurants in Seattle. The dining room has a beautiful view over the Puget Sound but it was surprisingly not crowed. I ordered the fried castelvetrano olives, a spicy Neapolitan-style pizza and a gnocchi dish. The olives were absolutely decadent, and the pizza came with a smoked mozzarella, which was delicious. The gnocchi was fresh and wonderful. The waitstaff were attentive, and overall the experience was lovely. I hope to return soon.",
        sentimentA="Positive",
        sentimentB="Negative",
        sentimentC="Neutral"
            )
prompt3 = PromptTemplate(input_variables=["context","taske","category1","category2","category3"], template=template3)
prompt_query3 = prompt3.format(
        task="Classify the given product description into given categories. Please output the category label in <output></output> tags.\n\nHere is the product description.",
        context="Safe, made from child-friendly materials with smooth edges. Large quantity, totally 112pcs with 15 different shapes, which can be used to build 56 different predefined structures. Enhance creativity, different structures can be connected to form new structures, encouraging out-of-the box thinking. Enhance child-parent bonding, parents can play with their children together to foster social skills.",
        category1="(1) Toys",
        category2="(2) Beauty and Health",
        category3="(3) Electronics"
        )


options = [{"desc":t1,"prompt_type":"Multiple-choice classification question", "prompt": "San Francisco, officially the City and County of San Francisco, is the commercial, financial, and cultural center of Northern California. The city proper is the fourth most populous city in California, with 808,437 residents, and the 17th most populous city in the United States as of 2022.\n\nWhat is the paragraph above about? Choose from the following:\n\nA city\nA person\nAn event", "prompt_query":prompt_query1,"height":210, "provider": "Amazon"},
            {"desc":t2,"prompt_type":"Sentiment analysis for text classification", "prompt": "The following is text from a restaurant review:\n\n“I finally got to check out Alessandro's Brilliant Pizza and it is now one of my favorite restaurants in Seattle. The dining room has a beautiful view over the Puget Sound but it was surprisingly not crowed. I ordered the fried castelvetrano olives, a spicy Neapolitan-style pizza and a gnocchi dish. The olives were absolutely decadent, and the pizza came with a smoked mozzarella, which was delicious. The gnocchi was fresh and wonderful. The waitstaff were attentive, and overall the experience was lovely. I hope to return soon.”\n\nTell me the sentiment of the restaurant review and categorize it as one of the following:\n\nPositive\nNegative\nNeutral", "prompt_query":prompt_query2, "height":300, "provider": "Amazon"},    
            {"desc":t3,"prompt_type":"Generate output enclosed in XML tags", "prompt": "Human: Classify the given product description into given categories. Please output the category label in <output></output> tags.\n\nHere is the product description.\n\n<text>\nSafe, made from child-friendly materials with smooth edges. Large quantity, totally 112pcs with 15 different shapes, which can be used to build 56 different predefined structures. Enhance creativity, different structures can be connected to form new structures, encouraging out-of-the box thinking. Enhance child-parent bonding, parents can play with their children together to foster social skills.\n</text>\n\nCategories are:\n(1) Toys\n(2) Beauty and Health\n(3) Electronics\n\nAssistant: ", "prompt_query":prompt_query3, "height":350, "provider": "Anthropic"},             
            ]

def update_options(item_num):
    st.session_state.prompt = options[item_num]["prompt"]
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]
    st.session_state.prompt_query = options[item_num]["prompt_query"]
    st.session_state.desc = options[item_num]["desc"]   

def load_options(item_num):
    st.button(f'{options[item_num]["prompt_type"]}', key=item_num, on_click=update_options, args=(item_num,))

row1_col1.markdown(st.session_state.desc)
with row1_col2:
    with st.container(border=True):
        provider = st.text_input('Provider',st.session_state.provider )
        model_id=st.text_input('model_id',getmodelId(st.session_state.provider))

def call_llm():
    # Instantiate LLM model
    llm = Bedrock(model_id=getmodelId(st.session_state.provider), client=bedrock_runtime, model_kwargs=getmodelparams(st.session_state.provider))
    #print(st.session_state.prompt_query)
    # Run LLM model
    response = llm.invoke(st.session_state.prompt_query)
    # Print results
    return st.info(response)

container = st.container(border=False)
    
with container:
    col1, col2, col3, col4 = st.columns([0.2,0.2,0.2,0.4])
    with col1:
        load_options(item_num=0)
    with col2:
        load_options(item_num=2)
    with col3:
        load_options(item_num=1)

with st.form("myform"):
    topic_text = st.text_area(":orange[Prompt:]", 
                              height = int(st.session_state.height),
                              key="prompt")
    submitted = st.form_submit_button("Submit", type="primary")
if topic_text and submitted:
    st.write("Answer")
    with st.spinner("Thinking..."):
        call_llm()
