import streamlit as st
import jsonlines
import json
import boto3
from PIL import Image
from io import BytesIO
from base64 import b64decode
from io import BytesIO
from random import randint
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as u_bedrock
import utils.stlib as stlib


params = {"model": "stability.stable-diffusion-xl-v1",
		#   "cfg_scale": 10,
		#   "seed": randint(10, 200000),
		#   "steps": 10,
		#   "width": 1024,
		#   "height": 1024,
		#  "style_preset":"3d-model"
  		# "size":"1024x1024"
		  }


def render_sdxl_image_code(templatePath, suffix):
	env = Environment(loader=FileSystemLoader('templates'))
	template = env.get_template(templatePath)
	output = template.render(
		prompt=st.session_state[suffix]['prompt'],
		height=int(st.session_state[suffix]['size'].split('x')[1]),
		width=int(st.session_state[suffix]['size'].split('x')[0]),
		cfg_scale=st.session_state[suffix]['cfg_scale'],
		seed=st.session_state[suffix]['seed'],
		steps=st.session_state[suffix]['steps'],
		negative_prompt=st.session_state[suffix]['negative_prompt'],
		model=st.session_state[suffix]['model'])
	return output


def update_parameters(suffix, **args):
	for key in args:
		st.session_state[suffix][key] = args[key]
	return st.session_state[suffix]


def load_jsonl(file_path):
	d = []
	with jsonlines.open(file_path) as reader:
		for obj in reader:
			d.append(obj)
	return d


def image_parameters(provider, suffix, index=0, region='us-east-1'):
	size = ["1024x1024", "1152x896", "1216x832", "1344x768", "1536x640", "640x1536", "768x1344", "832x1216", "896x1152"]
	st.subheader("Parameters")
	with st.container(border=True):
		models = u_bedrock.getmodelIds('Stability AI')
		model = st.selectbox(
			'model', models, index=models.index(u_bedrock.getmodelId(provider)))
		cfg_scale = st.slider('cfg_scale', value=st.session_state[suffix]['cfg_scale'], min_value=1, max_value=35, step=1)
		seed = st.number_input('seed', value=st.session_state[suffix]['seed'])
		selected_size = st.selectbox('size', size)
		width = int(selected_size.split('x')[0])
		height = int(selected_size.split('x')[1])
		steps = st.number_input('steps', value=st.session_state[suffix]['steps'], min_value=10, max_value=50, step=1)
		style_preset = st.selectbox('style', ["3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk", "origami","photographic", "pixel-art", "tile-texture"],index=14)
		new_params = {"model": model, 
				  "cfg_scale": cfg_scale, 
				  "seed": seed,
				  "steps": steps,
				  "width": width, 	
				  "height": height,
	  			  "style_preset":style_preset}
		col1, col2, col3 = st.columns([0.4,0.3,0.3])
		with col1:
			st.button(label = 'Tune Parameters', on_click=update_parameters, args=(suffix,), kwargs=(new_params))
		with col2:
			stlib.reset_session() 


# get the stringified request body for the InvokeModel API call
def get_sdxl_image_generation_request_body(prompt, negative_prompt, height, width, cfg_scale, seed,steps,style_preset):

	body = {"text_prompts": [
		{"text": prompt, "weight": 1},
		{"text": negative_prompt, "weight": -1}
	],
		"cfg_scale": float(cfg_scale),
		"seed": int(seed),
		"steps": int(steps),
		"height": int(height),
		"width": int(width),
		"style_preset": str(style_preset)
	}

	return json.dumps(body)


# get a BytesIO object from the sdxl Image Generator response
def get_sdxl_response_image(response):

	response = json.loads(response.get('body').read())

	images = response.get('artifacts')

	image = Image.open(BytesIO(b64decode(images[0].get('base64'))))
	image.save("images/generated_image.png")

	return image


# generate an image using Amazon sdxl Image Generator
def get_image_from_model(prompt, negative_prompt,model, height, width, cfg_scale, seed, steps,style_preset):

	bedrock = boto3.client(
		service_name='bedrock-runtime',
		region_name='us-east-1',
	)

	body = get_sdxl_image_generation_request_body(prompt=prompt, 
												  negative_prompt=negative_prompt,
												   height=height, 
												   width=width, 
												   cfg_scale=cfg_scale, 
												   seed=seed,
												   steps=steps,
			   										style_preset=style_preset
												   )

	response = bedrock.invoke_model(body=body, modelId=model,
									contentType="application/json", accept="application/json")

	output = get_sdxl_response_image(response)

	return output
