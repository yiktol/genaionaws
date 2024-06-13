import streamlit as st
import boto3
import json, sys
from datetime import datetime
from googlesearch import search
import requests
from bs4 import BeautifulSoup

import utils.helpers as helpers
bedrock = helpers.runtime_client()
helpers.set_page_config()

modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'

if "input" not in st.session_state:
    st.session_state.input = ""


class ToolsList:
    #Define our get_weather tool function...
    def get_weather(self, city, state):
        #st.write(city, state)
        result = f'Weather in {city}, {state} is 70F and clear skies.'
        st.write(f'Tool result: {result}')
        return result

class ToolsList3:
    #Define our get_weather tool function...
    def get_weather(self, city, state):
        #st.write(city, state)
        result = f'Weather in {city}, {state} is 70F and clear skies.'
        return result

    #Define our web_search tool function...
    def web_search(self, search_term):
        #st.write(f'{datetime.now().strftime("%H:%M:%S")} - Searching for {search_term} on Internet.')
        results = []
        response_list = []
        results.extend([r for r in search(search_term, 3, 'en')])
        for j in results:
            response = requests.get(j)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                response_list.append(soup.get_text().strip())
        response_text = ",".join(str(i) for i in response_list)
        #st.write(f'{datetime.now().strftime("%H:%M:%S")} - Search results: {response_text}')
        return response_text
    
#Define the configuration for our tool...
toolConfig = {'tools': [],
'toolChoice': {
    'auto': {},
    #'any': {},
    #'tool': {
    #    'name': 'get_weather'
    #}
    }
}
toolConfig['tools'].append({
        'toolSpec': {
            'name': 'get_weather',
            'description': 'Get weather of a location.',
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                            'description': 'City of the location'
                        },
                        'state': {
                            'type': 'string',
                            'description': 'State of the location'
                        }
                    },
                    'required': ['city', 'state']
                }
            }
        }
    })

toolConfig['tools'].append({
        'toolSpec': {
            'name': 'web_search',
            'description': 'Search a term in the public Internet. \
                Useful for getting up to date information.',
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {
                        'search_term': {
                            'type': 'string',
                            'description': 'Term to search in the Internet'
                        }
                    },
                    'required': ['search_term']
                }
            }
        }
    })

#Function for caling the Bedrock Converse API...
def converse_with_tools(messages, system='', toolConfig=toolConfig):
    response = bedrock.converse(
        modelId=modelId,
        system=system,
        messages=messages,
        toolConfig=toolConfig
    )
    return response


#Function for orchestrating the conversation flow...
def converse(prompt, system=''):
    #Add the initial prompt:
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "text": prompt
                }
            ]
        }
    )
    st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Initial prompt:\n{json.dumps(messages, indent=2)}")

    #Invoke the model the first time:
    output = converse_with_tools(messages, system)
    st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Output so far:\n{json.dumps(output['output'], indent=2, ensure_ascii=False)}")

    #Add the intermediate output to the prompt:
    messages.append(output['output']['message'])

    function_calling = next((c['toolUse'] for c in output['output']['message']['content'] if 'toolUse' in c), None)

    #Check if function calling is triggered:
    if function_calling:
        #Get the tool name and arguments:
        tool_name = function_calling['name']
        tool_args = function_calling['input'] or {}
        
        #Run the tool:
        st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Running ({tool_name}) tool...")
        tool_response = getattr(ToolsList(), tool_name)(**tool_args) or ""
        if tool_response:
            tool_status = 'success'
        else:
            tool_status = 'error'

        #Add the tool result to the prompt:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        'toolResult': {
                            'toolUseId':function_calling['toolUseId'],
                            'content': [
                                {
                                    "text": tool_response
                                }
                            ],
                            'status': tool_status
                        }
                    }
                ]
            }
        )
        #st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Messages so far:\n{json.dumps(messages, indent=2)}")

        #Invoke the model one more time:
        output = converse_with_tools(messages, system)
        st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Final output:\n{json.dumps(output['output'], indent=2, ensure_ascii=False)}\n")
    return


def converse_multi(prompt, system=''):
    #Add the initial prompt:
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "text": prompt
                }
            ]
        }
    )
    st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Initial prompt:\n{json.dumps(messages, indent=2)}")

    #Invoke the model the first time:
    output = converse_with_tools(messages, system)
    st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Output so far:\n{json.dumps(output['output'], indent=2, ensure_ascii=False)}")

    #Add the intermediate output to the prompt:
    messages.append(output['output']['message'])

    function_calling = next((c['toolUse'] for c in output['output']['message']['content'] if 'toolUse' in c), None)

    #Check if function calling is triggered:
    if function_calling:
        #Get the tool name and arguments:
        tool_name = function_calling['name']
        tool_args = function_calling['input'] or {}
        
        #Run the tool:
        st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Running ({tool_name}) tool...")
        tool_response = getattr(ToolsList3(), tool_name)(**tool_args)

        #Add the tool result to the prompt:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        'toolResult': {
                            'toolUseId':function_calling['toolUseId'],
                            'content': [
                                {
                                    "text": tool_response
                                }
                            ]
                        }
                    }
                ]
            }
        )

        #Invoke the model one more time:
        output = converse_with_tools(messages, system)
        st.write(f"\n{datetime.now().strftime('%H:%M:%S')} - Final output:\n{json.dumps(output['output'], indent=2, ensure_ascii=False)}\n")
    return

st.header("ReAct Prompting")
st.markdown("""ReAct is a general paradigm that combines reasoning and acting with LLMs. \
ReAct prompts LLMs to generate verbal reasoning traces and actions for a task.
            """)




options = [
    {"id":"0","prompt_type":"Weather Tool","prompt": "What is the weather like in Queens, NY?"},
    {"id":"1","prompt_type":"No Tool","prompt": "What is the capital of France?"},
    {"id":"2","prompt_type":"Web Search Tool","prompt": "In which team is Caitlin Clark playing in the WNBA in 2024?"},
    {"id":"2","prompt_type":"Web Search Tool","prompt": ""}
    ]

def update_options(item_num):
    st.session_state.input = options[item_num]["prompt"]
def load_options(item_num):
    st.write(f'Prompt: {options[item_num]["prompt"]}')
    st.button(f'Load Prompt', key=item_num, on_click=update_options, args=(item_num,))


# with st.container(border=True):
tabs = st.tabs(["Weather Tool", "No Tool","Web Search Tool","Test Your Own Prompt"])
for tab in tabs:
    with tab:
        item_num=tabs.index(tab)

        with st.form(f"form1-{item_num}"):
            input = st.text_area("Prompt:", value=options[item_num]["prompt"])
            submit = st.form_submit_button("Submit",type="primary")

        if submit:
            with st.spinner(f"Thinking..."):
                converse_multi(
                system = [{"text": "You're provided with a tool that can get the weather information for a specific locations 'get_weather'; \
                    only use the tool if required. Don't make reference to the tools in your final answer."}],
                prompt = input
        )
            



