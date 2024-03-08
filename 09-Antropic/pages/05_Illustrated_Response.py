import os, re
from IPython import display
from base64 import b64decode
from PIL import Image
from io import BytesIO
import json
import streamlit as st
from helpers import set_page_config, bedrock_runtime_client

set_page_config()
bedrock_runtime = bedrock_runtime_client()

with st.sidebar:
    with st.form(key ='Form1'):
        model = st.text_input('model', 'anthropic.claude-3-sonnet-20240229-v1:0', disabled=True)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.1)
        top_k=st.slider('top_k',min_value = 0, max_value = 300, value = 250, step = 1)
        top_p=st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('max_tokens',min_value = 50, max_value = 4096, value = 2048, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters') 
    

accept = 'application/json'
contentType = 'application/json'


image_gen_system_prompt = ("You are Claude, a helpful, honest, harmless AI assistant. "
"One special thing about this conversation is that you have access to an image generation API, "
"so you may create images for the user if they request you do so, or if you have an idea "
"for an image that seems especially pertinent or profound. However, it's also totally fine "
"to just respond to the human normally if that's what seems right! If you do want to generate an image, "
"write '<function_call>create_image(PROMPT)</function_call>', replacing PROMPT with a description of the image you want to create.")

image_gen_system_prompt += """

Here is some guidance for getting the best possible images:

<image_prompting_advice>
Rule 1. Make Your Stable Diffusion Prompts Clear, and Concise
Successful AI art generation in Stable Diffusion relies heavily on clear and precise prompts. It's essential to craft problem statements that are both straightforward and focused.

Clearly written prompts acts like a guide, pointing the AI towards the intended outcome. Specifically, crafting prompts involves choosing words that eliminate ambiguity and concentrate the AI's attention on producing relevant and striking images.
Conciseness in prompt writing is about being brief yet rich in content. This approach not only fits within the technical limits of AI systems but ensures each part of the prompt contributes meaningfully to the final image. Effective prompt creation involves boiling down complex ideas into their essence.
Prompt Example:
"Minimalist landscape, vast desert under a twilight sky."
This prompt exemplifies how a few well-chosen words can paint a vivid picture. The terms 'minimalist' and 'twilight sky' work together to set a specific mood and scene, demonstrating effective prompts creation with brevity.

Another Example:
"Futuristic cityscape, neon lights, and towering skyscrapers."
Here, the use of descriptive but concise language creates a detailed setting without overwhelming the AI. This example showcases the importance of balancing detail with succinctness in prompt structuring methods.

Rule 2. Use Detailed Subjects and Scenes to Make Your Stable Diffusion Prompts More Specific
Moving into detailed subject and scene description, the focus is on precision. Here, the use of text weights in prompts becomes important, allowing for emphasis on certain elements within the scene.

Detailing in a prompt should always serve a clear purpose, such as setting a mood, highlighting an aspect, or defining the setting. The difference between a vague and a detailed prompt can be stark, often leading to a much more impactful AI-generated image. Learning how to add layers of details without overwhelming the AI is crucial.
Scene setting is more than just describing physical attributes; it encompasses emotions and atmosphere as well. The aim is to provide prompts that are rich in context and imagery, resulting in more expressive AI art.
Prompt Example:
"Quiet seaside at dawn, gentle waves, seagulls in the distance."
In this prompt, each element adds a layer of detail, painting a serene picture. The words 'quiet', 'dawn', and 'gentle waves' work cohesively to create an immersive scene, showcasing the power of specific prompts crafting.

Another Example:
"Ancient forest, moss-covered trees, dappled sunlight filtering through leaves."
This prompt is rich in imagery and detail, guiding the AI to generate an image with depth and character. It illustrates how detailed prompts can lead to more nuanced and aesthetically pleasing results.

Rule 3. Contextualizing Your Prompts: Providing Rich Detail Without Confusion
In the intricate world of stable diffusion, the ability to contextualize prompts effectively sets apart the ordinary from the extraordinary. This part of the stable diffusion guide delves into the nuanced approach of incorporating rich details into prompts without leading to confusion, a pivotal aspect of the prompt engineering process.

Contextualizing prompts is akin to painting a picture with words. Each detail added layers depth and texture, making AI-generated images more lifelike and resonant. The art of specific prompts crafting lies in weaving details that are vivid yet coherent.
For example, when describing a scene, instead of merely stating: 
"a forest."
one might say,

"a sunlit forest with towering pines and a carpet of fallen autumn leaves."
Other Prompt Examples:
"Starry night, silhouette of mountains against a galaxy-filled sky."
This prompt offers a clear image while allowing room for the AI's interpretation, a key aspect of prompt optimization. The mention of 'starry night' and 'galaxy-filled sky' gives just enough context without dictating every aspect of the scene.

Rule 4. Do Not Overload Your Prompt Details
While detail is desirable, overloading prompts with excessive information can lead to ambiguous results. This section of the definitive prompt guide focuses on how to strike the perfect balance.

Descriptive Yet Compact: The challenge lies in being descriptive enough to guide the AI accurately, yet compact enough to avoid overwhelming it. For instance, a prompt like, 'A serene lake, reflecting the fiery hues of sunset, bordered by shadowy hills' paints a vivid picture without unnecessary verbosity.
Precision in language is key in this segment of the stable diffusion styles. It's about choosing the right words that convey the most with the least, a skill that is essential in prompt optimization.
For example, instead of using:
"a light wind that can barely be felt but heard"
You can make it shorter:

whispering breeze
More Prompt Examples:
Sample prompt: "Bustling marketplace at sunset, vibrant stalls, lively crowds."

By using descriptive yet straightforward language, this prompt sets a vivid scene of a marketplace without overcomplicating it. It's an example of how well-structured prompts can lead to dynamic and engaging AI art.
</image_prompting_advice>

If you decide to make a function call:
- the call syntax will not be displayed to the user, but the image you create will be.
- please place the call after your text response (if any)."""


def gen_image(prompt, height=1024, width=1024, num_samples=1):
    engine_id = "stability.stable-diffusion-xl"

    body = {"text_prompts":[{"text":prompt}],
    "cfg_scale":7,
    "steps":30}

    response = bedrock_runtime.invoke_model(body=json.dumps(body), modelId=engine_id, accept=accept, contentType=contentType)
    response = json.loads(response.get('body').read())
    images = response.get('artifacts')
    
    image = Image.open(BytesIO(b64decode(images[0].get('base64'))))
    image.save("generated_image.png")

    #print(response['artifacts'][0]['base64'])
   
    return response['artifacts'][0]['base64']

def parse_response_and_gen_image(claude_response):
    if "<function_call>" in claude_response:
        image_prompt = claude_response.split('<function_call>create_image(')[1].split(')</function_call>')[0].replace('"', '')
        base64 = gen_image(image_prompt)
    else:
        image_prompt, base64 = None, None

    function_free_claude_response = re.sub(r'<function_call>.*</function_call>', '', claude_response)
    # return the image_prompt too
    return (function_free_claude_response, image_prompt, base64)


def illustrator_claude(prompt):
    
    messages_list = [
        {"role": "user", "content": prompt}
    ]
    
    body = {"max_tokens": max_tokens_to_sample, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages_list,
            "system":image_gen_system_prompt}     
    
    response = bedrock_runtime.invoke_model(body=json.dumps(body), # Encode to bytes
                                    modelId=model, 
                                    accept=accept, 
                                    contentType=contentType)

    response_body =  json.loads(response.get('body').read())
    # print(response_body)

    return parse_response_and_gen_image(response_body["content"][0]["text"])

options = [{"prompt": "When & how did the Cretaceous period end?" },
           {"prompt": "What should I make for dinner? I have a bunch of potatoes and eggplant lying around. Gimme your best dish!"},
           {"prompt": "What would Albert Einstein look like if he were a 90s kid?"},
           ]


if "sample_prompt" not in st.session_state:
    st.session_state.sample_prompt = options[0]["prompt"]

def update_options(item_num):
    st.session_state.sample_prompt = options[item_num]["prompt"]

def load_options(item_num):    
    st.write("Prompt:",options[item_num]["prompt"])
    st.button("Load Prompt", key=item_num+1, on_click=update_options, args=(item_num,))  

col1, col2 = st.columns(2)
with col1:
    st.subheader("Prompt")
    with st.form("myform"):
        prompt_data = st.text_area(
            ":orange[Enter your prompt here:]",
            height = 100,
            key="sample_prompt")
        submit = st.form_submit_button("Submit", type="primary")
        
with col2:
    st.subheader('Prompt Examples:')
    container = st.container(border=True)    
    with container:
        tab1, tab2, tab3 = st.tabs(["Prompt1", "Prompt2", "Prompt3"])
        with tab1:
            load_options(item_num=0)
        with tab2:
            load_options(item_num=1)
        with tab3:
            load_options(item_num=2)

        
if submit:
    with st.spinner("Thinking..."):
        function_free_response_dino, image_prompt_dino, b64_dino = illustrator_claude(prompt_data)
        st.write(function_free_response_dino)
        st.image('generated_image.png', caption="Generated Image")
        st.write(image_prompt_dino)











