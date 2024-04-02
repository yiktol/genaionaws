import utils.helpers as helpers
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.jurassic as jurassic
import streamlit as st

stlib.set_page_config()

suffix = 'jurassic'
if suffix not in st.session_state:
		st.session_state[suffix] = {}


dataset = helpers.load_jsonl('data/jurassic.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(jurassic.params,suffix)

bedrock_runtime = bedrock.runtime_client()

text, code = st.columns([0.7, 0.3])

with code:
					
	with st.container(border=True):
		provider = st.selectbox('provider', ['AI21'])
		model = jurassic.modelId()
	with st.container(border=True):
		params = jurassic.tune_parameters()

with text:

	st.title("AI21")
	st.write("AI21's Jurassic family of leading LLMs to build generative AI-driven applications and services leveraging existing organizational data. Jurassic supports cross-industry use cases including long and short-form text generation, contextual question answering, summarization, and classification. Designed to follow natural language instructions, Jurassic is trained on a massive corpus of web text and supports six languages in addition to English. ")

	with st.expander("See Code"):
			st.code(jurassic.render_jurassic_code('jurassic.jinja',suffix), language="python")


	tab_names = [f"Prompt {question['id']}" for question in dataset]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,dataset):
		with tab:
			response = jurassic.prompt_box(content['id'],
								model=model,
								context=content['prompt'],height=content['height'],
								**params)
			
			if response:
				st.write("### Answer")
				st.info(response)