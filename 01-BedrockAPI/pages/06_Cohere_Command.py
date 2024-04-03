import streamlit as st
import utils.stlib as stlib
import utils.cohere as cohere

stlib.set_page_config()

suffix = 'cohere'
if suffix not in st.session_state:
	st.session_state[suffix] = {}


dataset = cohere.load_jsonl('data/cohere.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(cohere.params,suffix)

text, code = st.columns([0.7, 0.3])

with code:

	with st.container(border=True):
		provider = st.selectbox('provider', ['Cohere'])
		model = cohere.modelId()
		streaming = st.checkbox('Streaming',value=True)
		
	with st.container(border=True):
		params = cohere.tune_parameters()

with text:
	st.title('Cohere')
	st.write('Cohere models are text generation models for business use cases. Cohere models are trained on data that supports reliable business applications, like text generation, summarization, copywriting, dialogue, extraction, and question answering.')

	with st.expander("See Code"): 
		st.code(cohere.render_cohere_code('command.jinja',suffix),language="python")
		
	tab_names = [f"Prompt {question['id']}" for question in dataset]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,dataset):
		with tab:
			generations = cohere.prompt_box(content['id'],
								model=model,
								context=content['prompt'],height=content['height'],
								streaming=streaming,
								**params)
			
			if generations and not streaming:
				st.write("### Answer")
				for index, generation in enumerate(generations):

					st.success(f"Generation {index + 1}\n")
					st.info(f"Text:\n {generation['text']}\n")
					if 'likelihood' in generation:
						st.write(f"Likelihood:\n {generation['likelihood']}\n")
					
					st.info(f"Reason: {generation['finish_reason']}\n\n")
