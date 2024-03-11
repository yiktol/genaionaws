import textwrap
import json
from utils import get_models, set_page_config,bedrock_runtime_client
import utils.helpers as helpers
import streamlit as st

set_page_config()
def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.button(label='Clear Session', on_click=form_callback)

dataset = helpers.load_jsonl('utils/jurassic.jsonl')

helpers.initsessionkeys(dataset[0])


text, code = st.columns([0.6,0.4])


# code.subheader("Parameters")
# with code.form(key ='Form1'):
#       model = st.selectbox('model', get_models('AI21'),index=4)
#       temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.7, step = 0.1)
#       top_p=st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
#       max_tokens_to_sample=st.number_input('max_tokens_to_sample',min_value = 50, max_value = 4096, value = 300, step = 1)
#       submitted1 = st.form_submit_button(label = 'Tune Parameters') 

bedrock_runtime = bedrock_runtime_client()


xcode = f'''
import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
)

body = {{
  "prompt": \"{textwrap.shorten(st.session_state["prompt"],width=50,placeholder='...')}\"
  "maxTokens": {st.session_state['max_tokens']},
  "temperature": {st.session_state['temperature']},
  "topP": {st.session_state['top_p']},
  "stopSequences": [],
  "countPenalty": {{
    "scale": 0
  }},
  "presencePenalty": {{
    "scale": 0    
  }},
  "frequencyPenalty": {{
    "scale": 0
  }}
}}

modelId = '{st.session_state['model']}' 
accept = 'application/json'
contentType = 'application/json'

#Invoke the model
response = bedrock_runtime.invoke_model(
    body=json.dumps(body), 
    modelId=modelId, 
    accept=accept, 
    contentType=contentType
    )

response_body = json.loads(response.get('body').read())

print(response_body.get('completions')[0].get('data').get('text'))
'''

modelId = st.session_state['model']
accept = 'application/json'
contentType = 'application/json'


with text:
   
  st.title("AI21")
  st.write("AI21's Jurassic family of leading LLMs to build generative AI-driven applications and services leveraging existing organizational data. Jurassic supports cross-industry use cases including long and short-form text generation, contextual question answering, summarization, and classification. Designed to follow natural language instructions, Jurassic is trained on a massive corpus of web text and supports six languages in addition to English. ")

  with st.expander("See Code"):
      st.code(xcode,language="python")
  # Define prompt and model parameters

  with st.form("myform"):
    prompt_data = st.text_area(
        "Enter your prompt here:",
            height = st.session_state['height'],
            value = st.session_state["prompt"]
    )
    submit = st.form_submit_button("Submit", type='primary')

  body = {
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
  }
  if prompt_data and submit:
    with st.spinner("Generating..."):
      #Invoke the model
      response = bedrock_runtime.invoke_model(body=json.dumps(body), modelId=modelId, accept=accept, contentType=contentType)
      response_body = json.loads(response.get('body').read())

      #print(response_body.get('completions')[0].get('data').get('text'))

      st.write("### Answer")
      st.info(response_body.get('completions')[0].get('data').get('text'))
    

with code:
    helpers.tune_parameters('AI21',index=5)
    st.subheader('Prompt Examples:')   
    container2 = st.container(border=True) 
    with container2:
        helpers.create_tabs(dataset)
