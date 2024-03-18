import json
from utils import set_page_config, bedrock_runtime_client
import utils.helpers as helpers
import streamlit as st

set_page_config()

helpers.reset_session()

bedrock_runtime = bedrock_runtime_client(region='us-east-1')

prompt = """Q: What is an atom? 
A: An atom is a tiny particle that makes up everything. 

Q: Who is Alvan Muntz? 
A: ? 

Q: What is Kozar-09? 
A: ? 

Q: How many moons does Mars have? 
A: Two, Phobos and Deimos. 

Q: Who is Neto Beto Roberto? 
A:
"""

text, code = st.columns([0.6,0.4])

with code:
    with st.container(border=True):
        provider = st.selectbox('Provider',('Amazon','Anthropic','AI21','Cohere','Meta'))
        model_id=st.text_input('model_id',helpers.getmodelId(provider))

    with st.form(key ='form2'):
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
        top_p=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens_to_sample=st.number_input('maxTokenCount',min_value = 50, max_value = 4096, value = 1024, step = 1)
        submitted1 = st.form_submit_button(label = 'Tune Parameters') 

with text:
    st.title('Factuality')
    st.markdown("""LLMs have a tendency to generate responses that sounds coherent and convincing but can sometimes be made up. \
Improving prompts can help improve the model to generate more accurate/factual responses and reduce the likelihood to generate inconsistent and made up responses.

Some solutions might include:
- provide ground truth (e.g., related article paragraph or Wikipedia entry) as part of context to reduce the likelihood of the model producing made up text.
- configure the model to produce less diverse responses by decreasing the probability parameters and instructing it to admit (e.g., "I don't know") when it doesn't know the answer.
- provide in the prompt a combination of examples of questions and responses that it might know about and not know about
                """)

    with st.form('form1'):
        prompt = st.text_area(":orange[Prompt]", prompt, height=350)
        submit = st.form_submit_button("Submit",type='primary')
        
    if submit:
        with st.spinner("Thinking..."):
            output = helpers.invoke_model(
                client=bedrock_runtime_client(), 
                prompt=prompt, 
                model=model_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens_to_sample,
                )
            #print(output)
            st.write("Answer:")
            st.info(output)
        



