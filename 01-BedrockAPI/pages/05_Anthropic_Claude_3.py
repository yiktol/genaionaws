import streamlit as st
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.claude3 as claude3
from IPython.display import Image



stlib.set_page_config()

suffix = 'claude3'
if suffix not in st.session_state:
	st.session_state[suffix] = {}

dataset = claude3.load_jsonl('data/claude3.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(claude3.params,suffix)
  
text, code = st.columns([0.7, 0.3])

with code:
					
	with st.container(border=True):
		provider = st.selectbox('provider', ['Anthropic'])
		model = claude3.modelId()
		streaming = st.checkbox('Streaming',value=True)
	with st.container(border=True):
		params = claude3.tune_parameters()

with text:
	st.title("Anthropic Claude 3 (MultiModal)")
	st.write("""Anthropic's Claude family of models - Haiku, Sonnet, and Opus - allow customers to choose the exact combination of intelligence, speed, and cost that suits their business needs. \
Claude 3 Opus, the company's most capable model, has set a market standard on benchmarks. \
All of the latest Claude models have vision capabilities that enable them to process and analyze image data, meeting a growing demand for multimodal AI systems that can handle diverse data formats. While the family offers impressive performance across the board, Claude 3 Haiku is one of the most affordable and fastest options on the market for its intelligence category.""")

	with st.expander("See Code"): 
		st.code(claude3.render_claude_code('claude3.jinja',suffix),language="python")

	tab_names = [f"Prompt {question['id']}" for question in dataset]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,dataset):
		with tab:
			image_data, media_type = claude3.image_selector(content)
			if image_data:
				st.image(content['image'])
			if content['system']:
				system = content['system']
			else:
				system = None
			
			response = claude3.prompt_box(content['id'], model, prompt=content['prompt'], 
									system=system, 
									media_type=media_type,
									image_data=image_data,
									height=content['height'], 
									streaming=streaming,
									**params)
	
			# if response:
			# 	st.write("### Answer")
			# 	st.info(response)