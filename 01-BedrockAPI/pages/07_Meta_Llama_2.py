import streamlit as st
import utils.stlib as stlib
import utils.llama as llama

stlib.set_page_config()

suffix = 'llama2'
if suffix not in st.session_state:
	st.session_state[suffix] = {}


bedrock_runtime = bedrock.runtime_client()

dataset = llama.load_jsonl('data/meta.jsonl')

stlib.initsessionkeys(dataset[0], suffix)
stlib.initsessionkeys(llama.params, suffix)

text, code = st.columns([0.7, 0.3])

with code:
					
	with st.container(border=True):
		provider = st.selectbox('provider', ['Meta'])
		model = llama.modelId()
	with st.container(border=True):
		params = llama.tune_parameters()

with text:

	st.title("Meta")
	st.write("Llama is a family of large language models that uses publicly available data for training. These models are based on the transformer architecture, which allows it to process input sequences of arbitrary length and generate output sequences of variable length. One of the key features of Llama models is its ability to generate coherent and contextually relevant text. This is achieved through the use of attention mechanisms, which allow the model to focus on different parts of the input sequence as it generates output. Additionally, Llama models use a technique called “masked language modeling” to pre-train the model on a large corpus of text, which helps it learn to predict missing words in a sentence.")

	with st.expander("See Code"):
		st.code(llama.render_meta_code(
			'llama.jinja', suffix), language="python")

	tab_names = [f"Prompt {question['id']}" for question in dataset]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,dataset):
		with tab:
			response = llama.prompt_box(content['id'],
								model=model,
								context=content['prompt'],height=content['height'],
								**params)
			
			if response:
				st.write("### Answer")
				st.success(f"Generated Text: {response['generation']}")
				st.info(f"Prompt Token count:  {response['prompt_token_count']}")
				st.info(f"Generation Token count:  {response['generation_token_count']}")
				st.info(f"Stop reason:  {response['stop_reason']}")
