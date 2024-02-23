import streamlit as st

st.session_state.messages = []

st.set_page_config(
    page_title="Images",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Images")
st.markdown("""Easily generate compelling images by providing text prompts to pre-trained models. In the playground, enter a text prompt to get started.

Models:
- Titan Image Generator G1 is an image generation model. It generates images from text, and allows users to upload and edit an existing image. Users can edit an image with a text prompt (without a mask) or parts of an image with an image mask, or extend the boundaries of an image with outpainting. It can also generate variations of an image.    
- SDXL generates images of high quality in virtually any art style and is the best open model for photorealism. Distinct images can be prompted without having any particular ‘feel’ imparted by the model, ensuring absolute freedom of style. SDXL 1.0 is particularly well-tuned for vibrant and accurate colors, with better contrast, lighting, and shadows than its predecessor, all in native 1024x1024 resolution. In addition, SDXL can generate concepts that are notoriously difficult for image models to render, such as hands and text or spatially arranged compositions (e.g., a woman in the background chasing a dog in the foreground).     
         """)

url = 'https://static.us-east-1.prod.workshops.aws/public/c1f82486-e727-4b23-936c-4eca0be5506e/static/labs'

st.markdown(f"""_Example image transformations:_

| Outpainting | Extension | Inpainting | Variation |
| --- | --- | --- | --- |
| ![Lab screenshot]({url}/bedrock-image-background/robots.png) | ![Lab screenshot]({url}/bedrock-image-extension/image-original.png) | ![Lab screenshot]({url}/bedrock-image-masking/dinos.png) | ![Lab screenshot]({url}/bedrock-image-variation/orig.jpg) |
| ![Lab screenshot]({url}/bedrock-image-background/arena.jpg) | ![Lab screenshot]({url}/bedrock-image-extension/image-extended.png) | ![Lab screenshot]({url}/bedrock-image-masking/easter-in.jpg) | ![Lab screenshot]({url}/bedrock-image-variation/var.jpg) |
""")