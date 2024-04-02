import streamlit as st
import utils.helpers as helpers


helpers.set_page_config()

questions = [
    {"id": 1, "height": 250, "title": "Instruction should be placed at the end",
     "template": "",
     "instruction": "The question or instruction should be placed at the end of the prompt for best results\n Including the task description, instruction or question at the end aids the model determining which information it has to find. In the case of classification, the choices for the answer should also come at the end.\n In the following open-book question-answer example, the user has a specific question about the text. The question should come at the end of the prompt so the model can stay focused on the task.",
     "context": """Tensions increased after the 1911-1912 Italo-Turkish War demonstrated Ottoman weakness and led to the formation of the Balkan League, an alliance of Serbia, Bulgaria, Montenegro, and Greece. The League quickly overran most of the Ottomans' territory in the Balkans during the 1912-1913 First Balkan War, much to the surprise of outside observers.\n\nThe Serbian capture of ports on the Adriatic resulted in partial Austrian mobilization starting on 21 November 1912, including units along the Russian border in Galicia. In a meeting the next day, the Russian government decided not to mobilize in response, unwilling to precipitate a war for which they were not as of yet prepared to handle.\n\nWhich country captured ports?"""},
    {"id": 2, "height": 230, "title": "Use separator characters for API calls",
	"template": "Human: {Text}\n\\n {Question}\n\\n {Choice1} {Choice2} {Choice3}\n\\n\\n Assistant:",
     "instruction": "Separator characters such as :orange[\\n] can affect the performance of LLMs significantly. For Claude models, it's necessary to include newlines when formatting the API calls to obtain desired responses.\n The formatting should always follow: :orange[Human: {{Query Content}}\\n\\nAssistant:]. For Amazon Titan models, adding :orange[\\n] at the end of a prompt helps improve the performance of the model.\n For classification tasks or questions with answer options, you can also separate the answer options by :orange[\\n] for Titan models. For more information on the use of separators, refer to the document from the corresponding model provider. The following example is a template for a classification task.",
     "context": """Archimedes of Syracuse was an Ancient mathematician, physicist, engineer, astronomer, and inventor from the ancient city of Syracuse. Although few details of his life are known, he is regarded as one of the leading scientists in classical antiquity.\n\nWhat was Archimedes? Choose one of the options below.\n\na) astronomer\nb) farmer\nc) sailor."""},
    {"id": 3, "height": 150, "title": "Complex tasks: build toward the answer step by step",
     "template": "",
     "instruction": "LLM models can provide clear steps for certain tasks, and including the phrase. Think step-by-step to come up with the right answer can help produce the appropriate output",
     "context": """At a Halloween party, Jack gets 15 candies. Jack eats 5 candies. He wants to give each friend 5 candies. How many friends can receive candies?\n\nThink step-by-step to come up with the right answer."""},
	{"id": 4, "height": 100, "title": "Provide a default output",
     "template": "",
     "instruction": "A default output can help prevent LLMs from returning answers that sound like they could be correct, even if the model has low confidence.",
     "context": """Provide a proof of the Riemann hypothesis. If you don't know a proof, respond by saying \"I don't know.\""""},

]


suffix = 'Example1'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

text, code = st.columns([0.7, 0.3])

with code:

    with st.container(border=True):
        provider = st.selectbox('provider', helpers.list_providers)
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
            if content['template']:
                with st.expander("Template"):
                    st.markdown(content['template'])

            output = helpers.prompt_box(content['id'], provider,
                                        model,
                                        context=content['context'], height=content['height'],
                                        **params)

            if output:
                st.write("### Answer")
                st.info(output)
