import streamlit as st
import utils.bedrock as bedrock
import utils.stlib as stlib
import utils.titan_image as titan_image

stlib.set_page_config()

suffix = 'titan_image'
if suffix not in st.session_state:
	st.session_state[suffix] = {}

bedrock_runtime = bedrock.runtime_client()

dataset = titan_image.load_jsonl('data/titan_image.jsonl')

stlib.initsessionkeys(titan_image.params,suffix)
stlib.initsessionkeys(dataset[0],suffix)


text, code = st.columns([0.7, 0.3])


with code:
	 
	with st.container(border=True):
		provider = st.selectbox("Select a provider", ['Amazon']) 
		model = titan_image.image_model()
	with st.container(border=True):
		params = titan_image.image_parameters()
  
with text:

	st.title('Amazon Titan Image')
	st.write("""Titan Image Generator G1 is an image generation model. \
It generates images from text, and allows users to upload and edit an existing image. \
Users can edit an image with a text prompt (without a mask) or parts of an image with an image mask, or extend the boundaries of an image with outpainting. \
It can also generate variations of an image.""")

	with st.expander("See Code"):
		st.code(titan_image.render_titan_image_code('titan_image.jinja',suffix),language="python")

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
					image = titan_image.get_image_from_model(model, prompt_text,negative_prompt,**params)
				st.image(image)
