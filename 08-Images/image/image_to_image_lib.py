import os
import boto3
import json
import base64
from PIL import Image
from io import BytesIO
from helpers import bedrock_runtime_client
#

# session = boto3.Session(
#     profile_name=os.environ.get("BWB_PROFILE_NAME")
# ) #sets the profile name to use for AWS credentials

# bedrock = session.client(
#     service_name='bedrock-runtime', #creates a Bedrock client
#     region_name=os.environ.get("BWB_REGION_NAME"),
#     endpoint_url=os.environ.get("BWB_ENDPOINT_URL")
# ) 

bedrock = bedrock_runtime_client()

bedrock_model_id = 'stability.stable-diffusion-xl-v1'

#

def get_resized_image_io(image_bytes):
    image_io = BytesIO(image_bytes)
    image = Image.open(image_io)
    resized_image = image.resize((1024, 1024))
    
    resized_io = BytesIO()
    resized_image.save(resized_io, format=image.format)
    return resized_io

#

def prepare_image_for_endpoint(image_bytes):
    
    resized_io = get_resized_image_io(image_bytes)
    
    img_str = base64.b64encode(resized_io.getvalue()).decode("utf-8")
    
    return img_str


def get_stability_ai_request_body(prompt, image_str = None):
    #see https://platform.stability.ai/docs/features/api-parameters
    body = {"text_prompts": [ {"text": prompt } ], "cfg_scale": 8.0, "steps": 50, "seed": 123463446,"width":1024,"height":1024,"start_schedule":0.6}
    
    if image_str:
        body["init_image"] = image_str
    
    return json.dumps(body)

#

def get_stability_ai_response_image(response):

    response = json.loads(response.get('body').read())
    images = response.get('artifacts')
    
    image_data = base64.b64decode(images[0].get('base64'))
    
    return BytesIO(image_data)

#

def get_altered_image_from_model(prompt_content, image_bytes):
    
    image_str = prepare_image_for_endpoint(image_bytes)
    
    body = get_stability_ai_request_body(prompt_content, image_str)
    
    response = bedrock.invoke_model(body=body, modelId=bedrock_model_id, contentType="application/json", accept="application/json")
    
    output = get_stability_ai_response_image(response)
    image = Image.open(output)
    image.save("new_image.png")
    
    return output


def get_image_response(prompt_content): #text-to-text client function
    
   
    request_body = json.dumps({"text_prompts": 
                               [ {"text": prompt_content } ], #prompts to use
                               "seed": 121245125, #seed for the random number generator
                               "cfg_scale": 8.0, #how closely the model tries to match the prompt
                               "steps": 50,"width":1024,"height":1024 }) #number of diffusion steps to perform
    
    response = bedrock.invoke_model(body=request_body, modelId=bedrock_model_id) #call the Bedrock endpoint
    
    output = get_stability_ai_response_image(response) #convert the response payload to a BytesIO object for the client to consume
    image = Image.open(output)
    image.save("generated_image.png")
    return output
    