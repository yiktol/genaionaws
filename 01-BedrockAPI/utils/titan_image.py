import streamlit as st
import jsonlines
import json
import boto3
import base64
from jinja2 import Environment, FileSystemLoader
from io import BytesIO
from random import randint
import utils.bedrock as client
import utils.stlib as stlib


params = {
	"cfg_scale":8.0,
	"seed":randint(10,2147483646),
	"quality":"premium",
	"width":1024,
	"height":1024,
	"numberOfImages":1,
	"model":"amazon.titan-image-generator-v1",
	}


def load_jsonl(file_path):
	d = []
	with jsonlines.open(file_path) as reader:
		for obj in reader:
			d.append(obj)
	return d


def render_titan_image_code(templatePath,suffix):
	env = Environment(loader=FileSystemLoader('templates'))
	template = env.get_template(templatePath)
	output = template.render(
		prompt=st.session_state[suffix]['prompt'], 
		quality=st.session_state[suffix]['quality'], 
		height=st.session_state[suffix]['height'], 
		width=st.session_state[suffix]['width'],
		cfgScale=st.session_state[suffix]['cfg_scale'], 
		seed=st.session_state[suffix]['seed'],
		negative_prompt=st.session_state[suffix]['negative_prompt'], 
		numberOfImages=st.session_state[suffix]['numberOfImages'],
		model = st.session_state[suffix]['model'])
	return output

def image_parameters(provider, suffix, index=0):
	size = ["512x512","1024x1024","768x768","768x1152", "384x576","1152x768","576x384","768x1280","384x640","1280x768","640x384"]
	st.subheader("Parameters")
	with st.form("image-form"):
		models = client.getmodelIds(provider)
		model  = st.selectbox('model', models,index=index)
		cfg_scale= st.slider('cfg_scale',value = 8.0,min_value = 1.1, max_value = 10.0, step = 1.0, help="Specifies how strongly the generated image should adhere to the prompt.")
		seed=st.number_input('seed', value = None, help="Use to control and reproduce results. (min=0, max=2,147,483,646)")
		quality=st.radio('quality',["premium", "standard"], horizontal=True)
		selected_size=st.selectbox('size',size, index=1, help="The WxH of the image in pixels.")
		width = int(selected_size.split('x')[0])
		height = int(selected_size.split('x')[1])
		numberOfImages=st.selectbox('numberOfImages',[1], disabled=True)
  
		if seed is None:
			seed = randint(10, 2147483646)
		params = {"model":model ,"cfg_scale":cfg_scale, "seed":seed,"quality":quality,
					"width":width,"height":height,"numberOfImages":numberOfImages}
		col1, col2 = st.columns([0.5,0.5])
		with col1:
			st.form_submit_button(label = 'Tune Parameters', on_click=stlib.update_parameters, args=(suffix,), kwargs=(params))
		# with col2:
		# 	stlib.reset_session() 


#get the stringified request body for the InvokeModel API call
def get_titan_image_generation_request_body(prompt, negative_prompt,numberOfImages,quality, height, width,cfgScale,seed):
	
	body = { #create the JSON payload to pass to the InvokeModel API
		"taskType": "TEXT_IMAGE",
		"textToImageParams": {
			"text": prompt,
			"negativeText": negative_prompt
		},
		"imageGenerationConfig": {
			"numberOfImages": numberOfImages,  # Number of images to generate
			"quality": quality,
			"height": height,
			"width": width,
			"cfgScale": cfgScale,
			"seed": seed
		}
	}
	
	# if negative_prompt:
	#     body['textToImageParams']['negativeText'] = negative_prompt
	
	return json.dumps(body)


#get a BytesIO object from the Titan Image Generator response
def get_titan_response_image(response):

	response = json.loads(response.get('body').read())
	
	images = response.get('images')
	
	image_data = base64.b64decode(images[0])

	return BytesIO(image_data)


#generate an image using Amazon Titan Image Generator
def get_image_from_model(prompt_content, negative_prompt, numberOfImages, quality, height, width, cfgScale, seed):

	bedrock = boto3.client(
	service_name='bedrock-runtime',
	region_name='us-east-1', 
	)
	
	body = get_titan_image_generation_request_body(prompt=prompt_content, negative_prompt=negative_prompt,numberOfImages=numberOfImages,quality=quality, height=height, width=width,cfgScale=cfgScale,seed=seed)
	
	response = bedrock.invoke_model(body=body, modelId="amazon.titan-image-generator-v1", contentType="application/json", accept="application/json")
	
	output = get_titan_response_image(response)
	
	return output

