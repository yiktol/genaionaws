import json
import streamlit as st
from helpers import set_page_config, bedrock_runtime_client
import random

set_page_config()
bedrock_runtime = bedrock_runtime_client()

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.text_input('model', 'anthropic.claude-3-haiku-20240307-v1:0', disabled=True)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_k=st.slider('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        top_p=st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens',min_value = 50, max_value = 4096, value = 2048, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters') 


def moderate_text(user_text, guidelines):
    prompt_template = """
    You are a content moderation expert tasked with categorizing user-generated text based on the following guidelines:

    {guidelines}

    Here is the user-generated text to categorize:
    <user_text>{user_text}</user_text>

    Based on the guidelines above, classify this text as either ALLOW or BLOCK. Return nothing else.
    """

    # Format the prompt with the user text
    prompt = prompt_template.format(user_text=user_text, guidelines=guidelines)

    # Send the prompt to Claude and get the response
    body = {"max_tokens": max_tokens_to_sample, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}]}    

    accept = 'application/json'
    contentType = 'application/json'
    
    response = bedrock_runtime.invoke_model(body=json.dumps(body), # Encode to bytes
                                    modelId=model, 
                                    accept=accept, 
                                    contentType=contentType)

    response_body = json.loads(response.get('body').read())


    return response_body.get('content')[0]['text']


def performance(prompt, user_post):
   
    # Send the prompt to Claude and get the response
    body = {"max_tokens": max_tokens_to_sample, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt.format(user_post=user_post)}]  }

    accept = 'application/json'
    contentType = 'application/json'
    
    response = bedrock_runtime.invoke_model(body=json.dumps(body), # Encode to bytes
                                    modelId=model, 
                                    accept=accept, 
                                    contentType=contentType)

    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0]['text']



guidelines1 = '''BLOCK CATEGORY:
- Promoting violence, illegal activities, or hate speech
- Explicit sexual content
- Harmful misinformation or conspiracy theories

ALLOW CATEGORY:
- Most other content is allowed, as long as it is not explicitly disallowed
'''

guidelines2 = '''BLOCK CATEGORY:
- Content that is not related to rollercoasters, theme parks, or the amusement industry
- Explicit violence, hate speech, or illegal activities
- Spam, advertisements, or self-promotion

ALLOW CATEGORY:
- Discussions about rollercoaster designs, ride experiences, and park reviews
- Sharing news, rumors, or updates about new rollercoaster projects
- Respectful debates about the best rollercoasters, parks, or ride manufacturers
- Some mild profanity or crude language, as long as it is not directed at individuals
'''

cot_prompt = '''You are a content moderation expert tasked with categorizing user-generated text based on the following guidelines:

BLOCK CATEGORY:
- Content that is not related to rollercoasters, theme parks, or the amusement industry
- Explicit violence, hate speech, or illegal activities
- Spam, advertisements, or self-promotion

ALLOW CATEGORY:
- Discussions about rollercoaster designs, ride experiences, and park reviews
- Sharing news, rumors, or updates about new rollercoaster projects
- Respectful debates about the best rollercoasters, parks, or ride manufacturers
- Some mild profanity or crude language, as long as it is not directed at individuals

First, inside of <thinking> tags, identify any potentially concerning aspects of the post based on the guidelines below and consider whether those aspects are serious enough to block the post or not. Finally, classify this text as either ALLOW or BLOCK inside <output> tags. Return nothing else.

Given those instructions, here is the post to categorize:

<user_post>{user_post}</user_post>'''

user_post1 = "Introducing my new band - Coaster Shredders. Check us out on YouTube!!"

examples_prompt = '''You are a content moderation expert tasked with categorizing user-generated text based on the following guidelines:

BLOCK CATEGORY:
- Content that is not related to rollercoasters, theme parks, or the amusement industry
- Explicit violence, hate speech, or illegal activities
- Spam, advertisements, or self-promotion

ALLOW CATEGORY:
- Discussions about rollercoaster designs, ride experiences, and park reviews
- Sharing news, rumors, or updates about new rollercoaster projects
- Respectful debates about the best rollercoasters, parks, or ride manufacturers
- Some mild profanity or crude language, as long as it is not directed at individuals

Here are some examples:

<examples>

Text: I'm selling weight loss products, check my link to buy!
Category: BLOCK

Text: I hate my local park, the operations and customer service are terrible. I wish that place would just burn down.
Category: BLOCK

Text: Did anyone ride the new RMC raptor Trek Plummet 2 yet? I've heard it's insane!
Category: ALLOW

Text: Hercs > B&Ms. That's just facts, no cap! Arrow > Intamin for classic woodies too.
Category: ALLOW

</examples>\n
Given those examples, here is the user-generated text to categorize:
<user_text>{user_post}</user_text>

Based on the guidelines above, classify this text as either ALLOW or BLOCK. Return nothing else.'''

user_post2 = "Why Boomerang Coasters Ain't It (Don't @ Me)"


user_comments1 = [
    "This movie was great, I really enjoyed it. The main actor really killed it!",
    "Delete this post now or you better hide. I am coming after you and your family.",
    "Stay away from the 5G cellphones!! They are using 5G to control you.",
    "Thanks for the helpful information!",
]

user_comments2 = [
    "Top 10 Wildest Inversions on Steel Coasters",
    "My Review of the New RMC Raptor Coaster at Cedar Point",
    "Best Places to Buy Cheap Hiking Gear",
    "My Thoughts on the Latest Marvel Movie",
    "Rumor: Is Six Flags Planning a Giga Coaster for 2025?",
    
]

if "user_comment" not in st.session_state:
    st.session_state.user_comment = user_comments1[0]

def update_options(item_num,user_comment):
    st.session_state.user_comment = user_comment[item_num]

def load_options(item_num,user_comment):    
    st.write("Prompt:",user_comment[item_num])
    st.button("Load Prompt", key=random.randint(0,20000) , on_click=update_options, args=(item_num,user_comment))  



def sample(user_comment,guidelines,formkey):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prompt")
        with st.form(formkey):
            prompt_data = st.text_area(
                ":orange[Enter your prompt here:]",
                height = 50,
                value =st.session_state.user_comment)
            submit = st.form_submit_button("Submit", type="primary")

        if submit:
            st.write("Answer")
            with st.spinner("Classifying..."):
                classification = moderate_text(prompt_data, guidelines)
                #print(f"Classification: {classification}\n")
                if classification in ["ALLOW"]:
                    st.success(f"Classification: {classification}\n")
                else:
                    st.warning(f"Classification: {classification}\n")


    with col2:
        st.subheader("Guidelines")
        with st.container(border=True):
            st.markdown(guidelines)

        st.subheader('Prompt Examples:')
        container2 = st.container(border=True)    
        with container2:
            tab1, tab2, tab3, tab4 = st.tabs(["Prompt1", "Prompt2", "Prompt3", "Prompt4"])
            with tab1:
                load_options(item_num=0,user_comment=user_comment)
            with tab2:
                load_options(item_num=1,user_comment=user_comment)
            with tab3:
                load_options(item_num=2,user_comment=user_comment)
            with tab4:
                load_options(item_num=3,user_comment=user_comment)


def sample2(prompt,user_post,formkey):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prompt")
        with st.form(formkey):
            prompt_data = st.text_area(
                ":orange[Enter your prompt here:]",
                height = 50,
                value =user_post)
            submit = st.form_submit_button("Submit", type="primary")

        if submit:
            st.write("Answer")
            with st.spinner("Thinking..."):
                response = performance(prompt, prompt_data)
                st.info(response)

    with col2:
        st.subheader("Guidelines")
        with st.container(border=True):
            st.markdown(prompt)



tab1, tab2, tab3, tab4 = st.tabs(["Basic Approach","Moderate a Rollercoaster Enthusiast Forum","Improving Performance with Chain of Thought (CoT)","Improving Performance with Examples"])
with tab1:
    sample(user_comments1,guidelines1,formkey="form1")
with tab2:
    sample(user_comments2,guidelines2,formkey="form2")
with tab3:
    sample2(cot_prompt,user_post1,formkey="form3")
with tab4:
    sample2(examples_prompt,user_post2,formkey="form4")