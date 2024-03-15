import os
import re
import boto3
import json
import base64
import numpy as np
import seaborn as sns
from PIL import Image
from io import BytesIO
from scipy.spatial.distance import cdist


boto3_session = boto3.session.Session()
region_name = 'us-east-1'
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name,
)

def titan_image(
    payload:dict, 
    num_image:int=2, 
    cfg:float=10.0, 
    seed:int=2024
) -> list:

    body = json.dumps(
        {
            **payload,
            "imageGenerationConfig": {
                "numberOfImages": num_image,   # Number of images to be generated. Range: 1 to 5 
                "quality": "premium",          # Quality of generated images. Can be standard or premium.
                "height": 1024,                # Height of output image(s)
                "width": 1024,                 # Width of output image(s)
                "cfgScale": cfg,               # Scale for classifier-free guidance. Range: 1.0 (exclusive) to 10.0
                "seed": seed                   # The seed to use for re-producibility. Range: 0 to 214783647
            }
        }
    )

    response = bedrock_client.invoke_model(
        body=body, 
        modelId="amazon.titan-image-generator-v1", 
        accept="application/json", 
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    images = [
        Image.open(
            BytesIO(base64.b64decode(base64_image))
        ) for base64_image in response_body.get("images")
    ]

    return images

def extract_text(input_string):
    pattern = r"- (.*?)($|\n)"
    matches = re.findall(pattern, input_string)
    extracted_texts = [match[0] for match in matches]
    return extracted_texts

def embedding(
    image_path:str=None,  # maximum 2048 x 2048 pixels
    description:str=None, # English only and max input tokens 128
    dimension:int=1024,   # 1,024 (default), 384, 256
    model_id:str="amazon.titan-embed-image-v1"
):
    payload_body = {}
    embedding_config = {
        "embeddingConfig": { 
             "outputEmbeddingLength": dimension
         }
    }

    # You can specify either text or image or both
    if image_path:
        with open(image_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode('utf8')
        payload_body["inputImage"] = input_image
    if description:
        payload_body["inputText"] = description

    assert payload_body, "please provide either an image and/or a text description"
    print("\n".join(payload_body.keys()))

    response = bedrock_client.invoke_model(
        body=json.dumps({**payload_body, **embedding_config}), 
        modelId=model_id,
        accept="application/json", 
        contentType="application/json"
    )

    return json.loads(response.get("body").read())

def plot_similarity_heatmap(embeddings_a, embeddings_b):
    inner_product = np.inner(embeddings_a, embeddings_b)
    sns.set(font_scale=1.1)
    graph = sns.heatmap(
        inner_product,
        vmin=np.min(inner_product),
        vmax=1,
        cmap="OrRd",
    )

def search(query_emb:np.array, indexes:np.array, top_k:int=1):
    dist = cdist(query_emb, indexes, metric="cosine")
    return dist.argsort(axis=-1)[0,:top_k], np.sort(dist, axis=-1)[:top_k]

def multimodal_search(description:str,multimodal_embeddings:np.array,top_k:int,dimension:int=1024):
    query_emb = embedding(description=description, dimension=dimension)["embedding"]

    idx_returned, dist = search(
        np.array(query_emb)[None], 
        np.array(multimodal_embeddings),
        top_k=top_k,
    )
    
    return idx_returned

