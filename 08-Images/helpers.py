import streamlit as st
import json
import boto3
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm

def get_embedding(bedrock, text):
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    input = {
            'inputText': text
        }
    body=json.dumps(input)
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept,contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body['embedding']
    return embedding

def calculate_distance(v1, v2):
    distance = math.dist(v1, v2)
    return distance

def calculate_dot_product(v1, v2):
    similarity = dot(v1, v2)
    return similarity

def calculate_cosine_similarity(v1, v2):
    similarity = dot(v1, v2)/(norm(v1)*norm(v2))
    return similarity

def search(dataset, v):
    distance_array=[]
    for item in dataset:
        item['distance'] = calculate_distance(item['embedding'], v)
        distance_array.append(item['distance'])
    dataset.sort(key=lambda x: x['distance'])
    return [dataset[0]['text'],distance_array]



def classify(classes, v):
    distance_array=[]
    for item in classes:
        item['distance'] = calculate_distance(item['embedding'], v)
        distance_array.append(item['distance'])
    classes.sort(key=lambda x: x['distance'])
    return [classes[0]['name'],distance_array]
  
def find_outliers_by_count(dataset, count):
    # find the center of mass
    embeddings = []
    for item in dataset:
        embeddings.append(item['embedding'])
    center = np.mean(embeddings, axis=0)
    # calculate distance from center
    distances=[]
    for item in dataset:
        item['distance'] = calculate_distance(item['embedding'], center)
        distances.append(item['distance'])
    sd = np.std(embeddings)
    # sort the distances in reverse order
    dataset.sort(key=lambda x: x['distance'], reverse=True)
    # return N outliers
    return [dataset[0:count],distances]

def find_outliers_by_percentage(dataset, percent):
    # find the center of mass
    embeddings = []
    for item in dataset:
        embeddings.append(item['embedding'])
    center = np.mean(embeddings, axis=0)
    # calculate distance from center
    for item in dataset:
        item['distance'] = calculate_distance(item['embedding'], center)
    # sort the distances in reverse order
    dataset.sort(key=lambda x: x['distance'], reverse=True)
    # return top x% outliers
    total = len(dataset)
    count = math.floor(percent * total / 100)
    return dataset[0:count]  

def find_outliers_by_distance(dataset, percent):
    # find the center of mass
    embeddings = []
    for item in dataset:
        embeddings.append(item['embedding'])
    center = np.mean(embeddings, axis=0)
    # calculate distance from center
    for item in dataset:
        item['distance'] = calculate_distance(item['embedding'], center)
    # sort the distances in reverse order
    dataset.sort(key=lambda x: x['distance'], reverse=True)
    # return outliers beyond x% of max distance
    max_distance = dataset[0]['distance']
    min_distance = percent * max_distance / 100
    outliers = []
    for item in dataset:
        if item['distance'] >= min_distance:
            outliers.append(item)
    return outliers  
  

def invoke_model(client, prompt, model, 
    accept = 'application/json', content_type = 'application/json',
    max_tokens  = 512, temperature = 1.0, top_p = 1.0, top_k = 250, stop_sequences = [],
    count_penalty = 0, presence_penalty = 0, frequency_penalty = 0, return_likelihoods = 'NONE'):
    # default response
    output = ''
    # identify the model provider
    provider = model.split('.')[0] 
    # InvokeModel
    if (provider == 'anthropic'): 
        input = {
            'prompt': prompt,
            'max_tokens_to_sample': max_tokens, 
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'stop_sequences': stop_sequences
        }
        body=json.dumps(input)
        response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
        response_body = json.loads(response.get('body').read())
        output = response_body['completion']
    elif (provider == 'ai21'): 
        input = {
            'prompt': prompt, 
            'maxTokens': max_tokens,
            'temperature': temperature,
            'topP': top_p,
            'stopSequences': stop_sequences,
            'countPenalty': {'scale': count_penalty},
            'presencePenalty': {'scale': presence_penalty},
            'frequencyPenalty': {'scale': frequency_penalty}
        }
        body=json.dumps(input)
        response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
        response_body = json.loads(response.get('body').read())
        completions = response_body['completions']
        for part in completions:
            output = output + part['data']['text']
    elif (provider == 'amazon'): 
        input = {
            'inputText': prompt,
            'textGenerationConfig': {
                  'maxTokenCount': max_tokens,
                  'stopSequences': stop_sequences,
                  'temperature': temperature,
                  'topP': top_p
            }
        }
        body=json.dumps(input)
        response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
        response_body = json.loads(response.get('body').read())
        results = response_body['results']
        for result in results:
            output = output + result['outputText']
    elif (provider == 'cohere'): 
        input = {
            'prompt': prompt, 
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': top_k,
            'p': top_p,
            'stop_sequences': stop_sequences,
            'return_likelihoods': return_likelihoods
        }
        body=json.dumps(input)
        response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
        response_body = json.loads(response.get('body').read())
        results = response_body['generations']
        for result in results:
            output = output + result['text']
    elif (provider == 'meta'): 
        input = {
            'prompt': prompt,
            'max_gen_len': max_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
        body=json.dumps(input)
        response = client.invoke_model(body=body, modelId=model, accept=accept,contentType=content_type)
        response_body = json.loads(response.get('body').read())
        output = response_body['generation']
    # return
    return output

def set_page_config():
    st.set_page_config(
    page_title="Images",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

    
def bedrock_runtime_client():
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )
    return bedrock_runtime

def get_secret(secret_name):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    return get_secret_value_response['SecretString']
