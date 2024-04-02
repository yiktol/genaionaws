import streamlit as st
import utils.helpers as helpers
from langchain.prompts import PromptTemplate


helpers.set_page_config()

template1 = '''{context}\n
{question}? Choose from the following:\n
{choice1}
{choice2}
{choice3}
'''
template2 = '''The following is text from a {text_type}:\n
{context}\n
Tell me the sentiment of the {text_type} and categorize it as one of the following:\n
{sentimentA}
{sentimentB}
{sentimentC}
'''
template3 = '''{task}\n
<text>{context}</text>\n
Categories are:\n
{category1}
{category2}
{category3}\n
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

t1 = '''For text classification, the prompt includes a question with several possible choices for the answer, and the model must respond with the correct choice. Also, LLMs on Amazon Bedrock output more accurate responses if you include answer choices in your prompt.

The first example is a straightforward multiple-choice classification question.
'''
t2 = '''For text classification, the prompt includes a question with several possible choices for the answer, and the model must respond with the correct choice. Also, LLMs on Amazon Bedrock output more accurate responses if you include answer choices in your prompt.

Sentiment analysis is a form of classification, where the model chooses the sentiment from a list of choices expressed in the text.
'''
t3 = '''The following example uses Claude models to classify text. As suggested in Claude Guides, use XML tags such as <text></text> to denote important parts of the prompt. Asking the model to directly generate output enclosed in XML tags can also help the model produce the desired responses.
'''


t4 = '''
Given a prompt, LLMs on Amazon Bedrock can respond with a passage of original text that matches the description. Here is one example:
'''
t5 = '''
For text generation use cases, specifying detailed task requirements can work well. In the following example, we ask the model to generate response with exclamation points.
'''
t6 = '''
In the following example, a user prompts the model to take on the role of a specific person when generating the text. Notice how the signature reflects the role the model is taking on in the response.
'''
template4 = '''Please write a {Text_Category} in the voice of {Role}.'''
template5 = '''{Task_specification}\nPlease write a {Text_Category} in the voice of {Role}.'''
template6 = '''{Role_assumption}\n{Task_description}.'''

prompt4 = PromptTemplate(input_variables=["Text_Category","Role"], template=template4)
prompt_query4 = prompt4.format(Text_Category="email",Role="friend")
prompt5 = PromptTemplate(input_variables=["Text_Category","Role","Task_specification"], template=template5)
prompt_query5 = prompt5.format(Text_Category="email",Task_specification="Write text with exclamation points.",Role="friend")
prompt6 = PromptTemplate(input_variables=["Role_assumption","Task_description"], template=template6)
prompt_query6 = prompt6.format(Role_assumption="My name is Jack.",Task_description="Help me write a note expressing my gratitude to my parents for taking my son (their grandson) to the zoo. I miss my parents so much.")



questions = [
    {"id": 1, "height": 250, "title": "Multiple-choice classification",
     "instruction": t1,
     "context": prompt_query1},
    {"id": 2, "height": 300, "title": "Sentiment analysis",
     "instruction": t2,
     "context": prompt_query2},
    {"id": 3, "height": 350, "title": "Generate output enclosed in XML tags",
     "instruction": t3,
     "context": prompt_query3},
    {"id": 4, "height": 50, "title": "Text Category",
     "instruction": t4,
     "context": prompt_query4},
    {"id": 5, "height": 50, "title": "Task specification",
     "instruction": t5,
     "context": prompt_query5},
    {"id": 6, "height": 50, "title": "Role assumption",
     "instruction": t6,
     "context": prompt_query6},
]


suffix = 'Example1'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

text, code = st.columns([0.7, 0.3])

with code:

    with st.container(border=True):
        provider = st.selectbox('provider', helpers.list_providers, index=1)
        models = helpers.getmodelIds(provider)
        model = st.selectbox(
            'model', models, index=models.index(helpers.getmodelId(provider)))

    with st.container(border=True):
        params = helpers.tune_parameters(provider)

with text:

    tab_names = [question['title'] for question in questions]

    tabs = st.tabs(tab_names)

    for tab, content in zip(tabs, questions):
        with tab:
            st.markdown(content['instruction'])

            output = helpers.prompt_box(content['id'], provider,
                                        model,
                                        context=content['context'], height=content['height'],
                                        **params)

            if output:
                st.write("### Answer")
                st.info(output)
