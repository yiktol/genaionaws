import streamlit as st
import utils.helpers as helpers


helpers.set_page_config()

prompt1 = "Charles Mingus Jr. was an American jazz upright bassist, pianist, composer, bandleader, and author.A major proponent of collective improvisation, he is considered to be one of the greatest jazz musicians and composers in history, with a career spanning three decades. Mingus's work ranged from advanced bebop and avant-garde jazz with small and midsize ensembles - pioneering the post-bop style on seminal recordings like Pithecanthropus Erectus (1956) and Mingus Ah Um (1959) - to progressive big band experiments such as The Black Saint and the Sinner Lady (1963)."
prompt2 = "Extract names and years: the term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence. The synonym self-teaching computers was also used in this time period."
prompt3 = "Context: The NFL was formed in 1920 as the American Professional Football Association (APFA) before renaming itself the National Football League for the 1922 season. After initially determining champions through end-of-season standings, a playoff system was implemented in 1933 that culminated with the NFL Championship Game until 1966. Following an agreement to merge the NFL with the rival American Football League (AFL), the Super Bowl was first held in 1967 to determine a champion between the best teams from the two leagues and has remained as the final game of each NFL season since the merger was completed in 1970.\n\nQuestion: Based on the above context, when was the first Super Bowl?"


t1 = '''Add details about the constraints you would like to have on the output that the model should produce. \
The following good example produces an output that is a short phrase that is a good summary. \
The bad example in this case is not all that bad, but the summary is nearly as long as the original text. \
Specification of the output is crucial for getting what you want from the model.
'''
t2 = '''Here we give some additional examples from Claude and AI21 Jurassic models using output indicators. \
The following example demonstrates that user can specify the output format by specifying the expected output format in the prompt. \
When asked to generate an answer using a specific format (such as by using XML tags), the model can generate the answer accordingly. \
Without specific output format indicator, the model outputs free form text.'''
t3 = '''The following example shows a prompt and answer for the AI21 Jurassic model. \
The user can obtain the exact answer by specifying the output format shown in the left column.
'''


questions = [
    {"id": 1, "height": 200, "title": "With Output Indicator",
     "instruction": t1,
     "context": f"{prompt1}\n\nPlease summarize the above text in one phrase."},
    {"id": 2, "height": 200, "title": "No Output Indicator",
     "instruction": t1,
     "context": f"{prompt1}\n\nPlease summarize the above text."},
    {"id": 3, "height": 150, "title": "With Output Indicator",
     "instruction": t2,
     "context": f"{prompt2}\n\nPlease generate answer in <name></name> and <year></year> tags."},
	{"id": 4, "height": 100, "title": "No Output Indicator",
     "instruction": t2,
     "context": f"{prompt2}"},
    {"id": 5, "height": 250, "title": "With Output Indicator",
     "instruction": t3,
     "context": f"{prompt3}\n\nPlease only output the year."},
	{"id": 6, "height": 200, "title": "No Output Indicator",
     "instruction": t3,
     "context": f"{prompt3}"},
]


suffix = 'Example1'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

text, code = st.columns([0.7, 0.3])

with code:

    with st.container(border=True):
        provider = st.selectbox('provider', helpers.list_providers, index=3)
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
