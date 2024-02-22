import os
import boto3
import json
import base64
from io import BytesIO
from random import randint
from helpers import bedrock_runtime_client

#get the stringified request body for the InvokeModel API call
def get_titan_image_generation_request_body(prompt, negative_prompt=None):
    
    body = { #create the JSON payload to pass to the InvokeModel API
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,  # Number of images to generate
            "quality": "premium",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
            "seed": randint(0, 100000),  # Use a random seed
        },
    }
    
    if negative_prompt:
        body['textToImageParams']['negativeText'] = negative_prompt
    
    return json.dumps(body)


#get a BytesIO object from the Titan Image Generator response
def get_titan_response_image(response):

    response = json.loads(response.get('body').read())
    
    images = response.get('images')
    
    image_data = base64.b64decode(images[0])

    return BytesIO(image_data)


#generate an image using Amazon Titan Image Generator
def get_image_from_model(prompt_content, negative_prompt=None):
    # session = boto3.Session(
    #     profile_name=os.environ.get("BWB_PROFILE_NAME")
    # ) #sets the profile name to use for AWS credentials

    # bedrock = session.client(
    #     service_name='bedrock-runtime', #creates a Bedrock client
    #     region_name=os.environ.get("BWB_REGION_NAME"),
    #     endpoint_url=os.environ.get("BWB_ENDPOINT_URL")
    # ) 

    bedrock = bedrock_runtime_client()
    
    body = get_titan_image_generation_request_body(prompt_content, negative_prompt=negative_prompt)
    
    response = bedrock.invoke_model(body=body, modelId="amazon.titan-image-generator-v1", contentType="application/json", accept="application/json")
    
    output = get_titan_response_image(response)
    
    return output

