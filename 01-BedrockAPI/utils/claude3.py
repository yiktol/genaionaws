import streamlit as st
import jsonlines
import json
import boto3
from jinja2 import Environment, FileSystemLoader
import utils.bedrock as bedrock
import base64


bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime', region_name='us-east-1')


def load_jsonl(file_path):
    d = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            d.append(obj)
    return d


params = {"model": "anthropic.claude-3-sonnet-20240229-v1:0",
          "max_tokens": 1024,
          "temperature": 0.1,
          "top_k": 50,
          "top_p": 0.9,
          "stop_sequences": ["\n\nHuman"],
          }

accept = 'application/json'
content_type = 'application/json'


def getmodelIds_claude3():
    models = []
    available_models = bedrock.list_foundation_models()

    for model in available_models['modelSummaries']:
        if model['modelId'] in ['anthropic.claude-3-sonnet-20240229-v1:0:28k','anthropic.claude-3-sonnet-20240229-v1:0:200k','anthropic.claude-3-haiku-20240307-v1:0:48k','anthropic.claude-3-haiku-20240307-v1:0:200k']:
            continue
        elif "anthropic.claude-3" in model['modelId']:
            models.append(model['modelId'])

    return models


def modelId():
    models = getmodelIds_claude3()
    model = st.selectbox(
        'model', models, index=models.index("anthropic.claude-3-sonnet-20240229-v1:0"))

    return model


def render_claude_code(templatePath, suffix):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template(templatePath)
    output = template.render(
        prompt=st.session_state[suffix]['prompt'],
        max_tokens=st.session_state[suffix]['max_tokens'],
        temperature=st.session_state[suffix]['temperature'],
        top_p=st.session_state[suffix]['top_p'],
        top_k=st.session_state[suffix]['top_k'],
        model=st.session_state[suffix]['model'],
        stop_sequences=st.session_state[suffix]['stop_sequences']
    )
    return output


def claude_generic(input_prompt):
    prompt = f"""Human: {input_prompt}\n\nAssistant:"""
    return prompt


def update_parameters(suffix, **args):
    for key in args:
        st.session_state[suffix][key] = args[key]
    return st.session_state[suffix]


def tune_parameters():
    temperature = st.slider('temperature', min_value=0.0,
                            max_value=1.0, value=0.1, step=0.1)
    top_p = st.slider('top_p', min_value=0.0,
                      max_value=1.0, value=0.9, step=0.1)
    top_k = st.slider('top_k', min_value=0, max_value=100, value=50, step=1)
    max_tokens = st.number_input(
        'max_tokens', min_value=50, max_value=4096, value=1024, step=1)
    stop_sequences = st.text_input('stop_sequences', value="\n\nHuman")
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop_sequences": [stop_sequences],
        "max_tokens": max_tokens
    }

    return params


def image_selector(item):

    if item['image']:
        with open(item['image'], "rb") as image_file:
            binary_data = image_file.read()
            base_64_encoded_data = base64.b64encode(binary_data)
            base64_string = base_64_encoded_data.decode('utf-8')
        image_data = base64_string
    else:
        image_data = None

    if item['media_type']:
        media_type = item['media_type']
    else:
        media_type = None

    return image_data, media_type


def prompt_box(key, model, prompt, system=None, media_type=None, image_data=None, height=100, streaming=False, **params):
    response = ''
    system_prompt = None
    with st.form(f"form-{key}"):
        if system:
            system_prompt = st.text_area(
                ":orange[System Prompt:]",
                height=height,
                value=system
            )
        prompt_data = st.text_area(
            ":orange[User Prompt:]",
            height=height,
            value=prompt
        )
        submit = st.form_submit_button("Submit", type='primary')
    if submit:
        with st.spinner("Generating..."):
            if streaming:
                response = invoke_model_streaming(
                    prompt=prompt_data,
                    model=model,
                    system=system_prompt,
                    media_type=media_type,
                    image_data=image_data,
                    **params)
            else:
                response = invoke_model(
                    prompt=prompt_data,
                    model=model,
                    system=system_prompt,
                    media_type=media_type,
                    image_data=image_data,
                    **params)

    return response


def invoke_model(prompt, model,
                 system=None,
                 media_type=None,
                 image_data=None,
                 **params
                 ):
    output = ''

    if media_type is not None and image_data is not None:
        message_list = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": media_type, "data": image_data}}
                ]
            }
        ]
    else:
        message_list = [{"role": "user", "content": prompt}]

    input = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": message_list
    }

    if system:
        input['system'] = system

    input.update(params)

    body = json.dumps(input)
    response = bedrock_runtime.invoke_model(body=body,  # Encode to bytes
                                            modelId=model,
                                            accept=accept,
                                            contentType=content_type)

    response_body = json.loads(response.get('body').read())

    output = response_body.get('content')[0]['text']
    st.info(output)

    return output


def invoke_model_streaming(prompt, model,
                           system=None,
                           media_type=None,
                           image_data=None,
                           **params
                           ):
    output = ''

    if media_type is not None and image_data is not None:
        message_list = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": media_type, "data": image_data}}
                ]
            }
        ]
    else:
        message_list = [{"role": "user", "content": prompt}]

    input = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": message_list
    }

    if system:
        input['system'] = system

    input.update(params)

    body = json.dumps(input)
    response = bedrock_runtime.invoke_model_with_response_stream(body=body,  # Encode to bytes
                                                                 modelId=model,
                                                                 accept=accept,
                                                                 contentType=content_type)

    # response_body = json.loads(response.get('body').read())

    # output = response_body.get('content')[0]['text']
    # st.info(response)
    placeholder = st.empty()
    full_response = ''

    for event in response.get("body"):
        chunk = json.loads(event["chunk"]["bytes"])

        # if chunk['type'] == 'message_delta':
        #     print(f"\nStop reason: {chunk['delta']['stop_reason']}")
        #     print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
        #     print(f"Output tokens: {chunk['usage']['output_tokens']}")

        if chunk['type'] == 'content_block_delta':
            if chunk['delta']['type'] == 'text_delta':
                part = chunk['delta']['text']
                full_response += part
                placeholder.info(full_response)
            placeholder.info(full_response)

    return full_response
