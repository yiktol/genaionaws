import streamlit as st
import utils.stlib as stlib
import utils.claude2 as claude2

stlib.set_page_config()

suffix = 'claude2'
if suffix not in st.session_state:
	st.session_state[suffix] = {}


dataset = claude2.load_jsonl('data/anthropic.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(claude2.params,suffix)

text, code = st.columns([0.6,0.4])




text, code = st.columns([0.7, 0.3])

with code:
					
	with st.container(border=True):
		provider = st.selectbox('provider', ['Anthropic'])
		model = claude2.modelId()
		streaming = st.checkbox('Streaming',value=True)
	with st.container(border=True):
		params = claude2.tune_parameters()

with text:
	st.title("Anthropic Claude 2")
	st.write("""Anthropic offers the Claude family of large language models purpose built for conversations, 
			summarization, Q&A, workflow automation, coding and more. 
			Early customers report that Claude is much less likely to produce harmful outputs, 
			easier to converse with, and more steerable - so you can get your desired output with less effort. 
			Claude can also take direction on personality, tone, and behavior.""")

	with st.expander("See Code"): 
		st.code(claude2.render_claude_code('claude.jinja',suffix),language="python")

	tab_names = [f"Prompt {question['id']}" for question in dataset]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,dataset):
		with tab:
			response = claude2.prompt_box(content['id'],
								model=model,
								context=content['prompt'],height=content['height'],
								streaming=streaming,
								**params)
			
			# if response and not streaming:
			# 	st.write("### Answer")
			# 	# st.info(response)