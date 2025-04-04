import re
import boto3
import utils.helpers as helpers
from base64 import b64decode
from PIL import Image
from io import BytesIO
import json
import streamlit as st

st.set_page_config(
    page_title="Gen AI in Practice",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)
bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name='us-east-1' )


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


accept = 'application/json'
contentType = 'application/json'

def gen_image(prompt, height=1024, width=1024, num_samples=1):
    engine_id = "stability.stable-diffusion-xl-v1"

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


def illustrator_claude(prompt,**params):
    
    messages_list = [
        {"role": "user", "content": prompt}
    ]
    
    body = {"stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages_list,
            "system":image_gen_system_prompt}     
    body.update(params)
    
    response = bedrock_runtime.invoke_model(body=json.dumps(body), # Encode to bytes
                                    modelId=model, 
                                    accept=accept, 
                                    contentType=contentType)

    response_body =  json.loads(response.get('body').read())
    # print(response_body)

    return parse_response_and_gen_image(response_body["content"][0]["text"])


prompt1 = """Band Name: Look@Me Sunglass\n
Features:
- Frame: Tortoise acetate & burl wood
- Hinges: Stainless steel spring hinges
- Lenses: TAC Polarized - Grey
- UV Protection: 100% UVA/UVB - Polarized
- Specifications: Water & sweat resistant
- Included: Protective case & cleaning cloth

Write a description of a Sunglass with the following features.
"""

prompt2 = """Brand Name: Walkers\n
Walking Shoe Features:
- Achilles tendon protector. Reduces stress on the Achilles tendon by locking the shoe around the heel.
- Heel collar. Cushions the ankle and ensures proper fit.
- Upper. Holds the shoe on your foot and is usually made of leather, mesh or synthetic material. Mesh allows better ventilation and is lighter weight.
- Insole. Cushions and supports your foot and arch. Removable insoles can be laundered or taken out to dry between walking sessions.
- Gel, foam or air midsole. Helps cushion and reduce impact when your foot strikes the ground.
- Outsole. Makes contact with the ground. Grooves and treads can help maintain traction.
- Toe box. Provides space for the toes. A roomy and round toe box helps prevent calluses.

Write a product announcement of the walking shoes with the following features above.
"""

prompt3 = """Brand Name: GenAI T-Shirt\n
Product details:
- Logo: GenT
- Fabric type: 50% Cotton, 50% Polyester
- Care instructions: Machine Wash
- Sleeve type: Short Sleeve
- other details: Lightweight, Classic fit, Double-needle sleeve and bottom hem
- Size: M, L, XL, XXL

Create a social media post for a T-shirt with the following description. Write the word "Gen AI" in the center of the T-shirt.
"""

prompt4 = """Your task is to create a comprehensive design brief for a holistic brand identity based on the given specifications. The brand identity should encompass various elements such as suggestions for the brand name, logo, color palette, typography, visual style, tone of voice, and overall brand personality. Ensure that all elements work together harmoniously to create a cohesive and memorable brand experience that effectively communicates the brand's values, mission, and unique selling proposition to its target audience. Be detailed and comprehensive and provide enough specific details for someone to create a truly unique brand identity.\n\n Brand specs:\n This is a brand that focuses on creating high-quality, stylish clothing and accessories using eco-friendly materials and ethical production methods\n The brand targets environmentally conscious consumers aged 25-40 who value fashion, sustainability, and social responsibility.\n The brand identity should achieve the following goals:\n\n 1. Reflect the brand's commitment to sustainability, ethical practices, and environmental stewardship.\n 2. Appeal to the target audience by conveying a sense of style, quality, and trendiness.\n 3. Differentiate the brand from competitors in the sustainable fashion market.\n 4. Create a strong emotional connection with consumers and inspire them to make more environmentally friendly choices.\n"""

prompt5= """Your task is to generate creative, memorable, and marketable product names based on the provided description and keywords. The product names should be concise (2-4 words), evocative, and easily understood by the target audience. Avoid generic or overly literal names. Instead, aim to create a name that stands out, captures the essence of the product, and leaves a lasting impression.\n\nDescription: A noise-canceling, wireless, over-ear headphone with a 20-hour battery life and touch controls. Designed for audiophiles and frequent travelers.\n\nKeywords: immersive, comfortable, high-fidelity, long-lasting, convenient"""

prompt6 = """Your task is to suggest avant-garde fashion trends and styles tailored to the user's preferences. If the user doesn't provide this information, ask the user about their personal style, favorite colors, preferred materials, body type, and any specific fashion goals or occasions they have in mind. Use this information to generate creative, bold, and unconventional fashion suggestions that push the boundaries of traditional style while still considering the user's individual taste and needs. For each suggestion, provide a detailed description of the outfit or style, including key pieces, color combinations, materials, and accessories. Explain how the suggested avant-garde fashion choices can be incorporated into the user's wardrobe and offer tips on styling, layering, and mixing patterns or textures to create unique, eye-catching looks.\n\nPersonal style: Edgy, minimal, with a touch of androgyny\nFavorite colors: Black, white, and deep red\nPreferred materials: Leather, denim, and high-quality cotton\nBody type: Tall and lean\nFashion goals: To create a striking, fearless look for an art gallery opening"""

options = [{"id":1,"title":"Sunglass","prompt": prompt1,"height":300},
           {"id":2,"title":"Walking Shoe","prompt": prompt2,"height":350},
           {"id":3,"title":"T-Shirt","prompt": prompt3,"height":300},
           {"id":4,"title":"Brand builder","prompt": prompt4,"height":400},
           {"id":5,"title":"Product naming pro","prompt": prompt5,"height":250},
           {"id":6,"title":"Futuristic fashion advisor","prompt": prompt6,"height":350}
           
           ]


def prompt_box(prompt,key,height):
    with st.form(f"form-{key}"):
        prompt_data = st.text_area(
            ":orange[Enter your prompt here:]",
            height = height,
            value=prompt,
            key=key)
        submit = st.form_submit_button("Submit", type="primary")
    
    return submit, prompt_data
         
def generate_image(prompt_data,**params):
    with st.spinner("Generating..."):
        function_free_response_dino, image_prompt_dino, b64_dino = illustrator_claude(prompt_data,**params)
        st.write(function_free_response_dino)
        st.image('generated_image.png', caption="Generated Image")
        st.write(image_prompt_dino)
        
       
def getmodelIds_claude3(providername='Anthropic'):
	models =[]
	bedrock = boto3.client(service_name='bedrock',region_name='us-east-1' )
	available_models = bedrock.list_foundation_models()
	
	for model in available_models['modelSummaries']:
		if providername in model['providerName'] and "IMAGE" in model['inputModalities']:
			models.append(model['modelId'])
			
	return models
          
prompt_col, paramaters = st.columns([0.7,0.3])

with paramaters:
    with st.container(border=True):
        provider = st.selectbox('provider', ['Anthropic'])
        models = helpers.getmodelIds_claude3(providername='Anthropic')
        model = st.selectbox(
            'model', models, index=models.index("anthropic.claude-3-sonnet-20240229-v1:0"))

    with st.container(border=True):
        params = helpers.tune_parameters("Claude 3")


with prompt_col:

    tab_names = [option['title'] for option in options]

    tabs = st.tabs(tab_names)

    for tab, content in zip(tabs, options):
        with tab:
            submit, prompt_data = prompt_box(content["prompt"], content["id"], content["height"])
            if submit:
                generate_image(prompt_data,**params)













