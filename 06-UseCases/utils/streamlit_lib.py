import streamlit as st
import helpers

def tune_parameters():
    with st.container(border=True):
        provider = st.selectbox('Provider',('Amazon','Anthropic','AI21','Cohere','Meta','Mistral'))
        models = helpers.getmodelIds(provider)
        model_id=st.selectbox('model_id',models, index=models.index(helpers.getmodelId(provider)))

    with st.form(key ='form2'):
        temperature =st.slider('temperature',min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.1)
        top_p=st.slider('topP',min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)
        max_tokens=st.number_input('maxTokenCount',min_value = 50, max_value = 4096, value = 2048, step = 1)
        submit = st.form_submit_button(label = 'Tune Parameters') 
     
    match provider:
        case 'Anthropic':
            return temperature, top_p, max_tokens, model_id, submit, provider
        case 'AI21':
            return temperature, top_p, max_tokens, model_id, submit, provider
        case 'Amazon':
            return temperature, top_p, max_tokens, model_id, submit, provider
        case _:
            return temperature, top_p, max_tokens, model_id, submit, provider
        
        
        