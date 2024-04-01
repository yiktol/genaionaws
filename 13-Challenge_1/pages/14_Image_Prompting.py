import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()

if 'titan-image' not in st.session_state:
	st.session_state['titan-image'] = {}

if 'sdxl' not in st.session_state:
	st.session_state['sdxl'] = {}



task1 = "#### Ask the AI to generate an image for social media post."
context1 = """Organization Name: Green Thumb Org.

Tagline: Saving the Planet One tree at a time.

About the company: Green Thumb Organization is a non-profit organization founded in 2022. We provide service in educating companies and individual in the awareness of Global need for Sustainability.

Goals: The company main objective is to promote sustainability and the replanting of trees.

"""
output1 = """
"""

task2 = "#### Ask the AI to create a new SuperHero Character, define the superpowers you want."
context2 = """Create an Image of a new Super hero Character.
"""
output2 = """
"""

questions = [
	{"id":1,"task": task1, "context": context1, "output": output1},
	{"id":2,"task": task2, "context": context2, "output": output2}
]

text, code = st.columns([0.7, 0.3])


with code:
	 
	with st.container(border=True):
		provider = st.radio("Select a provider", ['Titan Image','Stability AI'], index=0, horizontal=True) 
		model = helpers.image_model(provider)
	with st.container(border=True):
		params = helpers.image_parameters(provider)
  
with text:

	tab_names = [f"Question {question['id']}" for question in questions]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,questions):
		with tab:
			st.markdown(content['task'])
			st.markdown(content['context'])
			with st.form(f"form-{content['id']}"):
				prompt_text = st.text_area("What you want to see in the image:",  height = 100, value = "", help="The prompt text")
				negative_prompt = st.text_area("What shoud not be in the image:", height = 100, value = "lowres, blurry", help="The negative prompt")
				generate_button = st.form_submit_button("Generate", type="primary")

			if generate_button:
				st.subheader("Result")
				with st.spinner("Drawing..."):
					image = helpers.generate_image(provider,model, prompt_text,negative_prompt,**params)
				st.image(image)
