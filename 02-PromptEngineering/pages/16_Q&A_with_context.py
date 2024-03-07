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
if "desc" not in st.session_state:
    st.session_state.desc = ""
if "question" not in st.session_state:
    st.session_state.question = ""
if "context" not in st.session_state:
    st.session_state.context = ""
if "instruction" not in st.session_state:
    st.session_state.instruction = ""
if "prompt_query" not in st.session_state:
    st.session_state.prompt_query = ""

row1_col1, row1_col2 = st.columns([0.7,0.3])


row1_col1.title("Question-answer")

t1 = '''
### Question-answer, with context

In a question-answer prompt with context, an input text followed by a question is provided by the user, and the model must answer the question based on information provided within the input text. Putting the question in the end after the text can help LLMs on Amazon Bedrock better answer the question. Model encouragement works for this use case as well.
'''
t2 = '''
### Question-answer, with context

When prompting Claude models, it's helpful to wrap the input text in XML tags. In the following example, the input text is enclosed in <text></text>.
'''
template1 = '''{Context}

{Question}'''
template2 = '''Human: {Instruction}\n
<text>
{Context}
</text>\n
{Question}

Assistant:'''

options = [{"desc":t1,"prompt_type":"Question-answer, with context 1", 
            "question": "Based on the information above, what species are red pandas closely related to?", 
            "context":"The red panda (Ailurus fulgens), also known as the lesser panda, is a small mammal native to the eastern Himalayas and southwestern China. It has dense reddish-brown fur with a black belly and legs, white-lined ears, a mostly white muzzle and a ringed tail. Its head-to-body length is 51-63.5 cm (20.1-25.0 in) with a 28-48.5 cm (11.0-19.1 in) tail, and it weighs between 3.2 and 15 kg (7.1 and 33.1 lb). It is well adapted to climbing due to its flexible joints and curved semi-retractile claws. The red panda was first formally described in 1825. The two currently recognized subspecies, the Himalayan and the Chinese red panda, genetically diverged about 250,000 years ago.\n\nThe red panda's place on the evolutionary tree has been debated, but modern genetic evidence places it in close affinity with raccoons, weasels, and skunks. It is not closely related to the giant panda, which is a bear, though both possess elongated wrist bones or \"false thumbs\" used for grasping bamboo. The evolutionary lineage of the red panda (Ailuridae) stretches back around 25 to 18 million years ago, as indicated by extinct fossil relatives found in Eurasia and North America.\n\nThe red panda inhabits coniferous forests as well as temperate broadleaf and mixed forests, favoring steep slopes with dense bamboo cover close to water sources. It is solitary and largely arboreal. It feeds mainly on bamboo shoots and leaves, but also on fruits and blossoms. Red pandas mate in early spring, with the females giving birth to litters of up to four cubs in summer. It is threatened by poaching as well as destruction and fragmentation of habitat due to deforestation. The species has been listed as Endangered on the IUCN Red List since 2015. It is protected in all range countries.",
            "instruction":"",
            "height":460, "provider": "Amazon"},
            {"desc":t2,"prompt_type":"Question-answer, with context 2", 
            "question": "Based on the text above, what songs did the Weekend play at the Super Bowl halftime show?", 
            "context":"On November 12, 2020, the selection of the Weekend to headline the show was announced; marking the first time a Canadian solo artist headlined the Super Bowl halftime show. When asked about preparations for the show, the Weekend stated, \"We've been really focusing on dialing in on the fans at home and making performances a cinematic experience, and we want to do that with the Super Bowl.\"\n\nThe performance featured a choir whose members were dressed in white and wore masks over their faces with glowing red eyes, and were standing within a backdrop of a neon cityscape. The performance opened with a white figure dressed the same as the choir being lowered into the backdrop where the choir was standing while singing “Call Out My Name\". At this time, the Weekend sat in a convertible against a skyline backdrop designed to resemble the Las Vegas Strip. For the next part of the performance, the backdrop then split open to reveal the Weekend, who then performed \"Starboy\", followed by \"The Hills\".\n\nNext, performing the song \"Can't Feel My Face\", the Weekend traveled through a labyrinth constructed behind the stage, joined by dancers dressed in red blazers and black neckties similar to his, but with their faces covered with bandages, in keeping with the aesthetic of his fourth studio album After Hours (2020). The dancers would wear these bandages throughout the performance. In the labyrinth section of the performance, camerawork was visually unsteady. The next songs performed were \"I Feel It Coming\", \"Save Your Tears\", and \"Earned It\". For the \"Earned It\" performance, the Weekend was accompanied by violinists. For the finale of the show, the Weekend took to the field of the stadium with his dancers to perform “Blinding Lights\". He and the dancers entered the field by performing \"House of Balloons / Glass Table Girls\". The performance ended with an array of fireworks.",
            "instruction":"Read the following text inside <text></text> XML tags, and then answer the question:",
            "height":590, "provider": "Anthropic"},]


def update_options(item_num,prompt_query):
    st.session_state.prompt_type = options[item_num]["prompt_type"]
    st.session_state.height = options[item_num]["height"]
    st.session_state.provider = options[item_num]["provider"]
    st.session_state.desc = options[item_num]["desc"]   
    st.session_state.question = options[item_num]["question"]
    st.session_state.context = options[item_num]["context"]
    st.session_state.instruction = options[item_num]["instruction"]
    st.session_state.prompt_query = prompt_query
    

def load_options(item_num,prompt_query):
    st.button(f'{options[item_num]["prompt_type"]}', key=item_num, on_click=update_options, args=(item_num,prompt_query))

prompt1 = PromptTemplate(input_variables=["Question","Context"], template=template1)
prompt_query1 = prompt1.format(Question=options[0]["question"],Context=options[0]["context"])
prompt2 = PromptTemplate(input_variables=["Question","Context","Instruction"], template=template2)
prompt_query2 = prompt2.format(Question=options[1]["question"],Context=options[1]["context"],Instruction=options[1]["instruction"])

row1_col1.markdown(st.session_state.desc)
with row1_col2:
    with st.container(border=True):
        provider = st.text_input('Provider',st.session_state.provider )
        model_id=st.text_input('model_id',getmodelId(st.session_state.provider))

def call_llm():
    # Instantiate LLM model
    llm = Bedrock(model_id=model_id,client=bedrock_runtime,model_kwargs=getmodelparams(provider))
    # Prompt
    
    # Run LLM model
    response = llm.invoke(st.session_state.prompt_query)
    # Print results
    return st.info(response)


container = st.container(border=False)
    
with container:
    col1, col2, col3= st.columns([0.2,0.2,0.6])
    with col1:
        load_options(item_num=0,prompt_query=prompt_query1)
    with col2:
        load_options(item_num=1, prompt_query=prompt_query2)

with st.form("myform"):
    text_prompt = st.text_area(":orange[Prompt:]", 
                              height = int(st.session_state.height),
                              disabled = False,
                              value = st.session_state.prompt_query)
    submitted = st.form_submit_button("Submit")
if submitted:
    st.write("Answer")
    with st.spinner("Thinking..."):
        call_llm()
