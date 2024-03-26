import streamlit as st
import utils.helpers as helpers
import utils.titan_image as titan_image
import utils.sdxl as sdxl

helpers.set_page_config()

suffix1 = 'titan-challenge'
if suffix1 not in st.session_state:
    st.session_state[suffix1] = {}

suffix2 = 'sdxl-challenge'
if suffix2 not in st.session_state:
    st.session_state[suffix2] = {}

titan_image.initsessionkeys(titan_image.params,suffix1)
sdxl.initsessionkeys(sdxl.params,suffix2)

task1 = "#### Ask the AI to generate an image for social media post."
context1 = """Organization Name: Green Thumb Org.

Tagline: Saving the Planet One tree at a time.

About the company: Green Thumb Organization is a non-profit organization founded in 2022. We provide service in educating companies and individual in the awareness of Global need for Sustainability.

Goals: The company main objective is to promote sustainability and the replanting of trees.

"""
output1 = """
"""

task2 = "#### Ask the AI to create a new SuperHero Character, define the superpowers you want."
context2 = """Create an Image of a new Super hero Character.
"""
output2 = """
"""

questions = [
    {"id":1,"task": task1, "context": context1, "output": output1},
    {"id":2,"task": task2, "context": context2, "output": output2}
]

text, code = st.columns([0.7, 0.3])


with code:
     
    with st.container(border=True):
        provider = st.radio("Select a provider", ['Amazon','Stability AI'], index=0, horizontal=True)    
    match provider:
        case 'Amazon':          
            titan_image.image_parameters('Amazon', suffix1)
        case 'Stability AI':
            sdxl.image_parameters('Stability AI', suffix2)

with text:

    tab1, tab2 = st.tabs(['Question 1','Question 2'])

    with tab1:
        st.markdown(task1)
        st.markdown(context1)
        with st.form("form1"):
            prompt_text = st.text_area("What you want to see in the image:",  height = 100, value = "", help="The prompt text")
            negative_prompt = st.text_area("What shoud not be in the image:", height = 100, value = "", help="The negative prompt")
            generate_button = st.form_submit_button("Generate", type="primary")

        if generate_button:
            st.subheader("Result")
            with st.spinner("Drawing..."):
   
                match provider:
                    case 'Amazon':   
                        generated_image = titan_image.get_image_from_model(
                            prompt_content = prompt_text, 
                            negative_prompt = negative_prompt,
                            numberOfImages = st.session_state[suffix1]['numberOfImages'],
                            quality = st.session_state[suffix1]['quality'], 
                            height = st.session_state[suffix1]['height'], 
                            width = st.session_state[suffix1]['width'], 
                            cfgScale = st.session_state[suffix1]['cfg_scale'], 
                            seed = st.session_state[suffix1]['seed']
                        )
                        
                    case 'Stability AI':
                        generated_image = sdxl.get_image_from_model(
                            prompt = prompt_text, 
                            negative_prompt = negative_prompt,
                            model=st.session_state[suffix2]['model'],
                            height = st.session_state[suffix2]['height'], 
                            width = st.session_state[suffix2]['width'], 
                            cfg_scale = st.session_state[suffix2]['cfg_scale'], 
                            seed = st.session_state[suffix2]['seed'],
                            steps = st.session_state[suffix2]['steps'],
                            style_preset = st.session_state[suffix2]['style_preset']
                            
                        )
            st.image(generated_image)

    with tab2:
        st.markdown(task2)
        st.markdown(context2)
        with st.form("form2"):
            prompt_text = st.text_area("What you want to see in the image:",  height = 100, value = "", help="The prompt text")
            negative_prompt = st.text_area("What shoud not be in the image:", height = 100, value = "", help="The negative prompt")
            generate_button2 = st.form_submit_button("Generate", type="primary")

        if generate_button2:
            st.subheader("Result")
            with st.spinner("Drawing..."):
   
                match provider:
                    case 'Amazon':   
                        generated_image = titan_image.get_image_from_model(
                            prompt_content = prompt_text, 
                            negative_prompt = negative_prompt,
                            numberOfImages = st.session_state[suffix1]['numberOfImages'],
                            quality = st.session_state[suffix1]['quality'], 
                            height = st.session_state[suffix1]['height'], 
                            width = st.session_state[suffix1]['width'], 
                            cfgScale = st.session_state[suffix1]['cfg_scale'], 
                            seed = st.session_state[suffix1]['seed']
                        )
                        
                    case 'Stability AI':
                        generated_image = sdxl.get_image_from_model(
                            prompt = prompt_text, 
                            negative_prompt = negative_prompt,
                            model=st.session_state[suffix2]['model'],
                            height = st.session_state[suffix2]['height'], 
                            width = st.session_state[suffix2]['width'], 
                            cfg_scale = st.session_state[suffix2]['cfg_scale'], 
                            seed = st.session_state[suffix2]['seed'],
                            steps = st.session_state[suffix2]['steps'],
                            style_preset = st.session_state[suffix2]['style_preset']
                            
                        )
            st.image(generated_image)

