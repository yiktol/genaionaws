import json
from utils import set_page_config, bedrock_runtime_client
import utils.helpers as helpers
import streamlit as st

set_page_config()

helpers.reset_session()

bedrock_runtime = bedrock_runtime_client(region='us-east-1')

prompt = """Q: The food here is delicious!
A: Positive 

Q: I'm so tired of this coursework.
A: Negative

Q: I can't believe I failed the exam.
A: Negative

Q: I had a great day today!
A: Positive 

Q: I hate this job.
A: Negative

Q: The service here is terrible.
A: Negative

Q: I'm so frustrated with my life.
A: Negative

Q: I never get a break.
A: Negative

Q: This meal tastes awful.
A: Negative

Q: I can't stand my boss.
A: Negative

Q: I feel something.
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
    st.title('Biases')
    st.markdown("""LLMs can produce problematic generations that can potentially be harmful and display biases that could deteriorate the performance of the model on downstream tasks. \
    Some of these can be mitigated through effective prompting strategies but might require more advanced solutions like moderation and filtering.
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
        
