import streamlit as st
import utils.helpers as helpers
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.titan_text as titan_text


stlib.set_page_config()

suffix = 'titan_text'
if suffix not in st.session_state:
	st.session_state[suffix] = {}

dataset = helpers.load_jsonl('data/titan_text.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(titan_text.params,suffix)

	   
text, code = st.columns([0.7, 0.3])

with code:

	with st.container(border=True):
		provider = st.selectbox('provider', ['Amazon'])
		model = titan_text.modelId()
		streaming = st.checkbox('Streaming')
		
	with st.container(border=True):
		params = titan_text.tune_parameters()

with text:
	st.title('Amazon Titan Text')
	st.write("""Titan Text models are generative LLMs for tasks such as summarization, text generation (for example, creating a blog post), classification, open-ended Q&A, and information extraction. They are also trained on many different programming languages as well as rich text format like tables, JSON and csv’s among others.""")

	with st.expander("See Code"):
		st.code(titan_text.render_titan_code('titan_text.jinja',suffix), language="python")
		
	tab_names = [f"Prompt {question['id']}" for question in dataset]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,dataset):
		with tab:
			
			output = titan_text.prompt_box(content['id'],
								model=model,
								context=content['prompt'],height=content['height'],
								streaming=streaming,
								**params)
			
			if output and not streaming:
			
				st.write("### Answer")
				st.info(output)



