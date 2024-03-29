import streamlit as st
import boto3
import utils.sdxl as sdxl
from random import randint

st.set_page_config(
	page_title="Mitigating Bias",
	page_icon=":rocket:",
	layout="wide",
	initial_sidebar_state="expanded",
)
suffix = 'sdxl'
if suffix not in st.session_state:
    st.session_state[suffix] = {}

def getmodelIds(providername):
    models =[]
    bedrock = boto3.client(service_name='bedrock',region_name='us-east-1' )
    available_models = bedrock.list_foundation_models()
    
    for model in available_models['modelSummaries']:
        if providername in model['providerName']:
            models.append(model['modelId'])
            
    return models

prompt1 = "a doctor in a hospital"
prompt2 = "a doctor in a hospital, inclusive of male and female"
prompt3 = "a doctor in a hospital, inclusive of male, female, and color"
prompt4 = "a nurse in a hospital, bias and discrimination against certain group of people"

negative_prompts = "bias,discriminatory,poorly rendered,poor background details,poorly drawn feature,disfigured features"

options = [{"id":1,"prompt": prompt1,"negative_prompts":negative_prompts,"p_height":50,"n_height":50},
           {"id":2,"prompt": prompt2,"negative_prompts":negative_prompts,"p_height":50,"n_height":50},
           {"id":3,"prompt": prompt3,"negative_prompts":negative_prompts,"p_height":50,"n_height":50},
           {"id":4,"prompt": prompt4,"negative_prompts":negative_prompts,"p_height":50,"n_height":50}
           ]


def prompt_box(prompt, negative_prompt, p_height, n_height, key):
    with st.form(f"form-{key}"):
        prompt_data = st.text_area("What you want to see in the image:",
                                    height=p_height,value = prompt, help="The prompt text")
        negative_prompt = st.text_area("What you don't want to see in the image:",
                                        height=n_height, value = negative_prompt, help="The negative prompt text")
        submit = st.form_submit_button("Submit", type='primary')
    
    return submit, prompt_data, negative_prompt
    
def generate_image(prompt_data,negative_prompt):
    with st.spinner("Generating..."):
        generated_image = sdxl.get_image_from_model(
                        prompt = prompt_data, 
                        negative_prompt = negative_prompt,
                        model=model,
                        height = height,
                        width = width,
                        cfg_scale = cfg_scale, 
                        seed = seed,
                        steps = steps,
                        style_preset = style_preset
                        
                    )
    st.image(generated_image)

text, code = st.columns([0.6, 0.4])

with code:

	size = ["1024x1024", "1152x896", "1216x832", "1344x768", "1536x640", "640x1536", "768x1344", "832x1216", "896x1152"]
	st.subheader("Parameters")
	with st.container(border=True):
		models = getmodelIds('Stability AI')
		model = st.selectbox(
			'model', models, index=models.index("stability.stable-diffusion-xl-v1"))
		cfg_scale = st.slider('cfg_scale', value=5, min_value=1, max_value=35, step=1)
		seed = st.number_input('seed', value=randint(0, 999999))
		selected_size = st.selectbox('size', size)
		width = int(selected_size.split('x')[0])
		height = int(selected_size.split('x')[1])
		steps = st.slider('steps', value=20, min_value=10, max_value=50, step=1)
		style_preset = st.selectbox('style', ["3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk", "origami","photographic", "pixel-art", "tile-texture"],index=14)


with text:
    tab1, tab2, tab3, tab4 = st.tabs(["Example1", "Example2", "Example3","Example4"])
    with tab1:
        submit, prompt_data, negative_prompt = prompt_box(options[0]["prompt"], options[0]["negative_prompts"], options[0]["p_height"], options[0]["n_height"], options[0]["id"])
        if submit:
            generate_image(prompt_data,negative_prompt)
    with tab2:
        submit, prompt_data, negative_prompt = prompt_box(options[1]["prompt"], options[1]["negative_prompts"], options[1]["p_height"], options[1]["n_height"], options[1]["id"] )
        if submit:
            generate_image(prompt_data,negative_prompt)
    with tab3:
        submit, prompt_data, negative_prompt = prompt_box(options[2]["prompt"], options[2]["negative_prompts"], options[2]["p_height"], options[2]["n_height"], options[2]["id"])
        if submit:
            generate_image(prompt_data,negative_prompt)
    with tab4:
        submit, prompt_data, negative_prompt = prompt_box(options[3]["prompt"], options[3]["negative_prompts"], options[3]["p_height"], options[3]["n_height"], options[3]["id"])
        if submit:
            generate_image(prompt_data, negative_prompt)


