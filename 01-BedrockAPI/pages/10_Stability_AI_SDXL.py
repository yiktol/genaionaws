import streamlit as st
import utils.stlib as stlib
import utils.sdxl as sdxl

stlib.set_page_config()

suffix = 'sdxl'
if suffix not in st.session_state:
	st.session_state[suffix] = {}


dataset = sdxl.load_jsonl('data/stabilityai.jsonl')

stlib.initsessionkeys(dataset[0],suffix)
stlib.initsessionkeys(sdxl.params,suffix)

text, code = st.columns([0.7, 0.3])


with code:
	 
	with st.container(border=True):
		provider = st.selectbox("Select a provider", ['Stability AI']) 
		model = sdxl.image_model()
	with st.container(border=True):
		params = sdxl.image_parameters()
  
with text:

	st.title("Stable Diffusion")
	st.write("Deep learning, text-to-image model used to generate detailed images conditioned on text descriptions, inpainting, outpainting, and generating image-to-image translations.")

	with st.expander("See Code"):
		st.code(sdxl.render_sdxl_image_code('sdxl.jinja',suffix), language="python")

	tab_names = [f"Image {question['id']}" for question in dataset]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,dataset):
		with tab:
			with st.form(f"form-{content['id']}"):
				prompt_text = st.text_area("What you want to see in the image:",  height = 100, value = content["prompt"], help="The prompt text")
				negative_prompt = st.text_area("What shoud not be in the image:", height = 100, value = content["negative_prompt"], help="The negative prompt")
				generate_button = st.form_submit_button("Generate", type="primary")

			if generate_button:
				st.subheader("Result")
				with st.spinner("Drawing..."):
					image = sdxl.get_image_from_model(prompt_text,negative_prompt,model,**params)
				st.image(image)
