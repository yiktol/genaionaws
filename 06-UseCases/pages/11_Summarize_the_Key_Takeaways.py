
import streamlit as st
import textwrap
from helpers import bedrock_runtime_client, set_page_config, invoke_model


set_page_config()


text, code = st.columns(2)

modelId = 'meta.llama2-13b-chat-v1'
prompt = """The seeds of a machine learning (ML) paradigm shift have existed for decades, but with the ready availability of scalable compute capacity, a massive proliferation of data, and the rapid advancement of ML technologies, customers across industries are transforming their businesses. \
Just recently, generative AI applications like ChatGPT have captured widespread attention and imagination. \
We are truly at an exciting inflection point in the widespread adoption of ML, and we believe most customer experiences and applications will be reinvented with generative AI. \
AI and ML have been a focus for Amazon for over 20 years, and many of the capabilities customers use with Amazon are driven by ML. \
Our e-commerce recommendations engine is driven by ML; the paths that optimize robotic picking routes in our fulfillment centers are driven by ML; and our supply chain, forecasting, and capacity planning are informed by ML. \
Prime Air (our drones) and the computer vision technology in Amazon Go (our physical retail experience that lets consumers select items off a shelf and leave the store without having to formally check out) use deep learning. \
Alexa, powered by more than 30 different ML systems, helps customers billions of times each week to manage smart homes, shop, get information and entertainment, and more. We have thousands of engineers at Amazon committed to ML, and it's a big part of our heritage, current ethos, and future. \
At AWS, we have played a key role in democratizing ML and making it accessible to anyone who wants to use it, including more than 100,000 customers of all sizes and industries. AWS has the broadest and deepest portfolio of AI and ML services at all three layers of the stack. \
We've invested and innovated to offer the most performant, scalable infrastructure for cost-effective ML training and inference; developed Amazon SageMaker, which is the easiest way for all developers to build, train, and deploy models; and launched a wide range of services that allow customers to add AI capabilities like image recognition, forecasting, and intelligent search to applications with a simple API call. \
This is why customers like Intuit, Thomson Reuters, AstraZeneca, Ferrari, Bundesliga, 3M, and BMW, as well as thousands of startups and government agencies around the world, are transforming themselves, their industries, and their missions with ML. \
We take the same democratizing approach to generative AI: we work to take these technologies out of the realm of research and experiments and extend their availability far beyond a handful of startups and large, well-funded tech companies. \
That's why today I'm excited to announce several new innovations that will make it easy and practical for our customers to use generative AI in their businesses.

Summarize the above text into three key takeaways. """

with code:
    
    with st.form(key ='form2'):
        # provider = st.text_input('Provider', modelId.split('.')[0],disabled=True)
        # model_id=st.text_input('model_id',modelId,disabled=True)
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.0, step = 0.1)
        top_p=st.slider('top_p',min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
        max_gen_len=st.number_input('max_gen_len',min_value = 50, max_value = 2048, value = 512, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters')        
        
    code_data = f"""import json
import boto3

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

modelId = \"{modelId}\"
accept = 'application/json'
contentType = 'application/json'
prompt = \"{textwrap.shorten(prompt,width=50,placeholder='...')}\"

input = {{
    'prompt': prompt,
    'max_gen_len': {max_gen_len},
    'temperature': {temperature},
    'top_p': {top_p}
}}

response = bedrock.invoke_model(
    body=json.dumps(input),
    modelId=modelId, 
    accept=accept,
    contentType=contentType
    )
    
response_body = json.loads(response.get('body').read())
results = response_body['generation']
print(results)
"""        
    
    st.code(code_data, language="python")




with text:

    # st.title("Extract Action Items")
    st.header("Summarize the Key Takeaways")
    st.write("In this example, we have a piece of long text. We want to summarize the text into three key takeaways.")
    with st.form('form1'):
        prompt = st.text_area(":orange[Prompt]", prompt, height=500)
        submit = st.form_submit_button("Summarize",type='primary')
        
    if submit:
        output = invoke_model(client=bedrock_runtime_client(), prompt=prompt, model=modelId,
                             temperature=temperature,
                             top_p=top_p,
                             max_tokens=max_gen_len
                             )
        st.write("Answer:")
        st.info(output)
    


