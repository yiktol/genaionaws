import streamlit as st
import utils.helpers as helpers


helpers.set_page_config()


t = '''In a question-answer prompt without context, the model must answer the question with its internal knowledge without using any context or document.
'''
t1 = '''In a question-answer prompt with context, an input text followed by a question is provided by the user, and the model must answer the question based on information provided within the input text. Putting the question in the end after the text can help LLMs on Amazon Bedrock better answer the question. Model encouragement works for this use case as well.
'''
t2 = '''When prompting Claude models, it's helpful to wrap the input text in XML tags. In the following example, the input text is enclosed in <text></text>.
'''

prompt1 = """The red panda (Ailurus fulgens), also known as the lesser panda, is a small mammal native to the eastern Himalayas and southwestern China. It has dense reddish-brown fur with a black belly and legs, white-lined ears, a mostly white muzzle and a ringed tail. Its head-to-body length is 51-63.5 cm (20.1-25.0 in) with a 28-48.5 cm (11.0-19.1 in) tail, and it weighs between 3.2 and 15 kg (7.1 and 33.1 lb). It is well adapted to climbing due to its flexible joints and curved semi-retractile claws. The red panda was first formally described in 1825. The two currently recognized subspecies, the Himalayan and the Chinese red panda, genetically diverged about 250,000 years ago.\n\nThe red panda's place on the evolutionary tree has been debated, but modern genetic evidence places it in close affinity with raccoons, weasels, and skunks. It is not closely related to the giant panda, which is a bear, though both possess elongated wrist bones or \"false thumbs\" used for grasping bamboo. The evolutionary lineage of the red panda (Ailuridae) stretches back around 25 to 18 million years ago, as indicated by extinct fossil relatives found in Eurasia and North America.\n\nThe red panda inhabits coniferous forests as well as temperate broadleaf and mixed forests, favoring steep slopes with dense bamboo cover close to water sources. It is solitary and largely arboreal. It feeds mainly on bamboo shoots and leaves, but also on fruits and blossoms. Red pandas mate in early spring, with the females giving birth to litters of up to four cubs in summer. It is threatened by poaching as well as destruction and fragmentation of habitat due to deforestation. The species has been listed as Endangered on the IUCN Red List since 2015. It is protected in all range countries.\n\nBased on the information above, what species are red pandas closely related to?"""
prompt2 = """Read the following text inside <text></text> XML tags, and then answer the question:\n\n\
<text>
On November 12, 2020, the selection of the Weekend to headline the show was announced; marking the first time a Canadian solo artist headlined the Super Bowl halftime show. When asked about preparations for the show, the Weekend stated, \"We've been really focusing on dialing in on the fans at home and making performances a cinematic experience, and we want to do that with the Super Bowl.\"\n\nThe performance featured a choir whose members were dressed in white and wore masks over their faces with glowing red eyes, and were standing within a backdrop of a neon cityscape. The performance opened with a white figure dressed the same as the choir being lowered into the backdrop where the choir was standing while singing “Call Out My Name\". At this time, the Weekend sat in a convertible against a skyline backdrop designed to resemble the Las Vegas Strip. For the next part of the performance, the backdrop then split open to reveal the Weekend, who then performed \"Starboy\", followed by \"The Hills\".\n\nNext, performing the song \"Can't Feel My Face\", the Weekend traveled through a labyrinth constructed behind the stage, joined by dancers dressed in red blazers and black neckties similar to his, but with their faces covered with bandages, in keeping with the aesthetic of his fourth studio album After Hours (2020). The dancers would wear these bandages throughout the performance. In the labyrinth section of the performance, camerawork was visually unsteady. The next songs performed were \"I Feel It Coming\", \"Save Your Tears\", and \"Earned It\". For the \"Earned It\" performance, the Weekend was accompanied by violinists. For the finale of the show, the Weekend took to the field of the stadium with his dancers to perform “Blinding Lights\". He and the dancers entered the field by performing \"House of Balloons / Glass Table Girls\". The performance ended with an array of fireworks.
</text>\n\n\
Based on the text above, what songs did the Weekend play at the Super Bowl halftime show?
"""

t3 = '''For complex reasoning tasks or problems that requires logical thinking, we can ask the model to make logical deductions and explain its answers.
'''

prompt3 = """The barber is the \"one who shaves all those, and those only, who do not shave themselves\".\nDoes the barber shave himself? Why is this a paradox?\n\n\
Please output the answer and then explain your answer.
"""

questions = [
    {"id": 1, "height": 100, "title": "Q&A",
     "instruction": t,
     "context": "What is Robert Frost's \"Stopping by the woods on a snowy evening\" about metaphorically?"},
    {"id": 2, "height": 100, "title": "Q&A,Encouragement",
     "instruction": t,
     "context": "You are excellent at answering questions, and it makes you happy when you provide the correct answer.\n\nWhat planet in the solar system is most likely to host life?"},
    {"id": 3, "height": 150, "title": "Q&A,Encouragement,Constraint",
     "instruction": t,
     "context": "You feel rewarded by helping people learn more about climate change.\n\nCould you please explain what climate change is?\n\nAssume your audience is composed of high school students."},
	{"id": 4, "height": 450, "title": "Q&A with Context",
     "instruction": t1,
     "context": prompt1},
    {"id": 5, "height": 550, "title": "Q&A with Context",
     "instruction": t2,
     "context": prompt2},
    {"id": 6, "height": 150, "title": "Reasoning/logical thinking",
     "instruction": t3,
     "context": prompt3},
]


suffix = 'Example3'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

text, code = st.columns([0.7, 0.3])

with code:

    with st.container(border=True):
        provider = st.selectbox('provider', helpers.list_providers, index=6)
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
