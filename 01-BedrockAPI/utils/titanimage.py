import json
import boto3
import base64
from io import BytesIO
from random import randint
from utils import bedrock_runtime_client, getmodelId


bedrock = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
    )

prompt = "Blue backpack on a table."
negative_prompt = ""

body = { 
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt,
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "quality": "premium",
        "height": 512,
        "width": 512,
        "cfgScale": 8.0,
        "seed": randint(0, 100000), 
    },
}

if negative_prompt:
    body['textToImageParams']['negativeText'] = negative_prompt

response = bedrock.invoke_model(
    body=body, 
    modelId="amazon.titan-image-generator-v1", 
    contentType="application/json", 
    accept="application/json")

response = json.loads(response.get('body').read())

images = response.get('images')

image_data = base64.b64decode(images[0])

image = BytesIO(image_data)