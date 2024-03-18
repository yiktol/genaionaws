
import streamlit as st
import textwrap
from helpers import bedrock_runtime_client, set_page_config, invoke_model, getmodelId


set_page_config()


text, code = st.columns([0.6,0.4])

modelId = 'ai21.j2-ultra'
prompt = """Meet Carbon Maps, a new French startup that raised $4.3 million (€4 million) just a few weeks after its inception. The company is building a software-as-a-service platform for the food industry so that they can track the environmental impact of each of their products in their lineup. The platform can be used as a basis for eco ratings. \
While there are quite a few carbon accounting startups like Greenly, Sweep, Persefoni and Watershed, Carbon Maps isn't an exact competitor as it doesn't calculate a company's carbon emissions as a whole. It doesn't focus on carbon emissions exclusively either. Carbon Maps focuses on the food industry and evaluates the environmental impact of products — not companies. \
Co-founded by Patrick Asdaghi, Jérémie Wainstain and Estelle Huynh, the company managed to raise a seed round with Breega and Samaipata — these two VC firms already invested in Asdaghi's previous startup, FoodChéri. \
FoodChéri is a full-stack food delivery company that designs its own meals and sells them directly to end customers with an important focus on healthy food. It also operates Seazon, a sister company for batch deliveries. The startup was acquired by Sodexo a few years ago. \
“On the day that I left, I started working on food and health projects again,” Asdaghi told me. “I wanted to make an impact, so I started moving up the supply chain and looking at agriculture.” \
And the good news is that Asdaghi isn't the only one looking at the supply chain of the food industry. In France, some companies started working on an eco-score with a public agency (ADEME) overseeing the project. It's a life cycle assessment that leads to a letter rating from A to E. \
While very few brands put these letters on their labels, chances are companies that have good ratings will use the eco-score as a selling point in the coming years. \
But these ratings could become even more widespread as regulation is still evolving. The European Union is even working on a standard — the Product Environmental Footprint (PEF). European countries can then create their own scoring systems based on these European criteria, meaning that food companies will need good data on their supply chains. \
“The key element in the new eco-score that's coming up is that there will be some differences within a product category because ingredients and farming methods are different,” Asdaghi said. “It's going to take into consideration the carbon impact, but also biodiversity, water consumption and animal welfare.” \
For instance, when you look at ground beef, it's extremely important to know whether farmers are using soy from Brazil or grass to feed cattle. \
“We don't want to create the ratings. We want to create the tools that help with calculations — a sort of SAP,” Asdaghi said. \
So far, Carbon Maps is working with two companies on pilot programs as it's going to require a ton of work to cover each vertical in the food industry. The startup creates models with as many criteria as possible to calculate the impact of each criteria. It uses data from standardized sources like GHG Protocol, IPCC, ISO 14040 and 14044. \
The company targets food brands because they design the recipes and select their suppliers. Eventually, Carbon Maps hopes that everybody across the supply chain is going to use its platform in one way or another. \
“You can't have a true climate strategy if you don't have some collaboration across the chain,” Asdaghi said. \

## 

Summarize the above text in 5 bullets."""

   
with code:
    with st.container(border=True):
        provider = st.selectbox('Provider',('Amazon','Anthropic','AI21','Cohere','Meta'))
        model_id=st.text_input('model_id',getmodelId(provider))
    
    with st.form(key ='form2'):
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
        topP=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 1.0, step = 0.1)
        maxTokens=st.number_input('maxTokens',min_value = 50, max_value = 4096, value = 2048, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters')    

    code_data = f"""import json
import boto3

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
    )

modelId = 'ai21.j2-ultra'
accept = 'application/json'
contentType = 'application/json'

prompt= \"{textwrap.shorten(prompt,width=50,placeholder='...')}\"

input = {{
    'prompt':prompt, 
    'maxTokens': {maxTokens},
    'temperature': {temperature},
    'topP': {topP},
    'stopSequences': [],
    'countPenalty': {{'scale': 0}},
    'presencePenalty': {{'scale': 0}},
    'frequencyPenalty': {{'scale': 0}}
        }}
        
response = bedrock.invoke_model(
    body=json.dumps(input),
    modelId=modelId, 
    accept=accept,
    contentType=contentType
    )
    
response_body = json.loads(response.get('body').read())
completions = response_body['completions']

for part in completions:
    print(part['data']['text'])

"""
        
    with st.expander("Show Code"):
        st.code(code_data, language="python")

with text:

    # st.title("Extract Action Items")
    st.header("Article Summarization")
    st.write("In this example, we want to create a summary of this very long article with 5 bullet points:")
    with st.form('form1'):
        prompt = st.text_area(":orange[Article]", prompt, height=500)
        submit = st.form_submit_button("Summarize",type='primary')
        
    if submit:
        with st.spinner("Thinking..."):
            output = invoke_model(client=bedrock_runtime_client(), 
                                prompt=prompt, 
                                model=model_id,
                                temperature=temperature,
                                top_p=topP,
                                max_tokens=maxTokens,)
            st.write("Answer:")
            st.info(output)