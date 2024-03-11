import textwrap
import json
from utils import get_models, set_page_config, bedrock_runtime_client
import utils.helpers as helpers
import streamlit as st

set_page_config()


def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]


st.sidebar.button(label='Clear Session', on_click=form_callback)

dataset = helpers.load_jsonl('utils/jurassic.jsonl')

helpers.initsessionkeys(dataset[0])

text, code = st.columns([0.6, 0.4])

bedrock_runtime = bedrock_runtime_client()

xcode = f'''import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
)

body = json.dumps({{
    "prompt": \"Human: {textwrap.shorten(st.session_state["prompt"],width=50,placeholder='...')}\n\nAssistant:\",
    "maxTokens": {st.session_state['max_tokens']},
    "temperature": {st.session_state['temperature']},
    "topP": {st.session_state['top_p']},
    "stopSequences": [],
    "countPenalty": {{"scale": 0}},
    "presencePenalty": {{"scale": 0}},
    "frequencyPenalty": {{"scale": 0}}
  }})

#Invoke the model
response = bedrock_runtime.invoke_model(
    body=body, 
    modelId='{st.session_state['model']}', 
    accept='application/json', 
    contentType='application/json'
    )

response_body = json.loads(response.get('body').read())
print(response_body.get('completions')[0].get('data').get('text'))
'''

with text:

  st.title("AI21")
  st.write("AI21's Jurassic family of leading LLMs to build generative AI-driven applications and services leveraging existing organizational data. Jurassic supports cross-industry use cases including long and short-form text generation, contextual question answering, summarization, and classification. Designed to follow natural language instructions, Jurassic is trained on a massive corpus of web text and supports six languages in addition to English. ")

  with st.expander("See Code"):
      st.code(xcode, language="python")
  # Define prompt and model parameters

  with st.form("myform"):
    prompt_data = st.text_area(
        "Enter your prompt here:",
        height=st.session_state['height'],
        value=st.session_state["prompt"]
    )
    submit = st.form_submit_button("Submit", type='primary')

  body = json.dumps({
      "prompt": prompt_data,
      "maxTokens": st.session_state['max_tokens'],
      "temperature": st.session_state['temperature'],
      "topP": st.session_state['top_p'],
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
  })

  if prompt_data and submit:
    with st.spinner("Generating..."):
      # Invoke the model
      response = bedrock_runtime.invoke_model(
          body=body, modelId=st.session_state['model'], accept='application/json', contentType='application/json')
      response_body = json.loads(response.get('body').read())

      st.write("### Answer")
      st.info(response_body.get('completions')[0].get('data').get('text'))

with code:
  
    helpers.tune_parameters('AI21', index=5)
    st.subheader('Prompt Examples:')
    container2 = st.container(border=True)
    with container2:
        helpers.create_tabs(dataset)
