from botocore.config import Config
from botocore.exceptions import ClientError
import json
from PIL import Image
from io import BytesIO
from base64 import b64decode
import boto3
import streamlit as st
from utils import bedrock_runtime_client, set_page_config
import utils.helpers as helpers


set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.button(label='Reset Session', on_click=form_callback)

bedrock_runtime = bedrock_runtime_client()

dataset = helpers.load_jsonl('utils/stabilityai.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6,0.4])

xcode = f'''import json
from PIL import Image
from io import BytesIO
import base64
from base64 import b64encode
from base64 import b64decode
import boto3

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
)

body = json.dumps({{"text_prompts":[{{"text":\"{st.session_state["prompt"]}\"}}],
        "cfg_scale":{st.session_state['cfg_scale']},
        "seed":{st.session_state['seed']},
        "steps":{st.session_state['steps']}
        }})

response = bedrock_runtime.invoke_model(
    body=body, 
        modelId='{st.session_state["model"]}', 
        accept='application/json', 
        contentType='application/json'
    )
    
response = json.loads(response.get('body').read())

images = response.get('artifacts')
image = Image.open(BytesIO(b64decode(images[0].get('base64'))))
image.save("generated_image.png")
'''


with text:
    st.title("Stable Diffusion")
    st.write("Deep learning, text-to-image model used to generate detailed images conditioned on text descriptions, inpainting, outpainting, and generating image-to-image translations.")

    with st.expander("See Code"):
        st.code(xcode,language="python")
        
    # Define prompt and model parameters
    with st.form("myform"):
        prompt_data = st.text_area("What you want to see in the image:",  height = st.session_state['height'], key = "prompt", help="The prompt text")
        submit = st.form_submit_button("Submit", type='primary')
        
        body = {"text_prompts":[{"text":prompt_data}],
        "cfg_scale":st.session_state['cfg_scale'],
        "seed":st.session_state['seed'],
        "steps":st.session_state['steps']}
        
        modelId = st.session_state["model"]
        accept = 'application/json'
        contentType = 'application/json'
                
    if prompt_data and submit:
        with st.spinner("Drawing..."):
            response = bedrock_runtime.invoke_model(body=json.dumps(body), modelId=modelId, accept=accept, contentType=contentType)
            response = json.loads(response.get('body').read())
            images = response.get('artifacts')

            image = Image.open(BytesIO(b64decode(images[0].get('base64'))))
            image.save("generated_image.png")

            st.write("### Answer")
            st.image(image)


with code:

    helpers.image_parameters("Stability AI", index=1,region='us-east-1')

    st.subheader('Prompt Examples:')   
    with st.container(border=True):
        helpers.create_tabs(dataset)


