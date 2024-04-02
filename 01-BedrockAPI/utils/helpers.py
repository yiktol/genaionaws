import streamlit as st
import jsonlines
import json
import utils.titan_text as titan
import utils.claude2 as claude2
import utils.llama as llama
import utils.mistral as mistral
import utils.cohere as cohere
import utils.jurassic as jurassic
import utils.claude3 as claude3
import utils.titan_image as titan_image
import utils.sdxl as sdxl
import base64
from io import BytesIO
from random import randint

from utils import bedrock_runtime_client

def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d



def getmodelparams(providername):
    model_mapping = {
        "Amazon" : {
            "maxTokenCount": 1024,
            "stopSequences": [], 
            "temperature": 0.1,
            "topP": 0.9
            },
        "Anthropic" : {
            "max_tokens_to_sample": 1024,
            "temperature": 0.1,
            "top_k": 50,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman"],
            },
        "AI21" : {
            "maxTokens": 1024,
            "temperature": 0.1,
            "topP": 0.9,
            "stopSequences": [],
            "countPenalty": {
            "scale": 0
            },
            "presencePenalty": {
            "scale": 0    
            },
            "frequencyPenalty": {
            "scale": 0
            }
        },
        "Cohere": {
            "max_tokens": 1024,
            "temperature": 0.1,
            "p": 0.9,
            "k": 50,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        },
        "Meta":{ 
            'max_gen_len': 1024,
            'top_p': 0.9,
            'temperature': 0.1
        }
    }
    
    return model_mapping[providername]

def prompt_box(key, provider, model, context=None, height=100, **params):
    response = ''
    with st.container(border=True):
        prompt = st.text_area("Enter your prompt here", value=context,
                              height=height,
                              key=f"Q{key}")
        submit = st.button("Submit", type="primary", key=f"S{key}")

    if submit:
        if context is not None:
            prompt = context + "\n\n" + prompt
            match provider:
                case "Amazon":
                    prompt = titan_generic(prompt)
                case "Anthropic":
                    prompt = claude_generic(prompt)
                case "Meta":
                    prompt = llama2_generic(prompt)
                case "Mistral":
                    prompt = mistral_generic(prompt)
                case _:
                    prompt = prompt

        with st.spinner("Generating..."):
            response = invoke_model(
                runtime_client(),
                prompt,
                model=model,
                **params)

    return response





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

# dataset = load_jsonl('mistral.jsonl')
# initsessionkeys(dataset[0])
# update_options(dataset,item_num=0)


