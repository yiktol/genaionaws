from langchain_community.llms import Bedrock
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()

def form_callback():
    for key in st.session_state.keys():
        del st.session_state[key]
        
st.sidebar.button(label='Clear Session Data', on_click=form_callback)



row1_col1, row1_col2 = st.columns([0.7,0.3])
row2_col1 = st.columns(1)

row1_col1.title("Use separator characters")

t = '''
### Use separator characters for API calls

Separator characters such as :orange[\\n] can affect the performance of LLMs significantly. For Claude models, it's necessary to include newlines when formatting the API calls to obtain desired responses.\n
The formatting should always follow: :orange[\\n\\nHuman: {{Query Content}}\\n\\nAssistant:]. For Amazon Titan models, adding :orange[\\n] at the end of a prompt helps improve the performance of the model.\n
For classification tasks or questions with answer options, you can also separate the answer options by :orange[\\n] for Titan models. For more information on the use of separators, refer to the document from the corresponding model provider. The following example is a template for a classification task.
'''
row1_col1.markdown(t)
with row1_col2:
    with st.container(border=True):
        provider = st.selectbox('Provider',('Amazon','Anthropic'), index=1, disabled=True)
        model_id=st.text_input('model_id',helpers.getmodelId(provider))

    st.write("**:orange[Template:]**")
    template = '''
    Human: {Text}\\n

    {Question}\\n

    {Choice1}
    {Choice2}
    {Choice3}\\n

    Assistant:
    '''

    st.code(template, language='None')


#Create the connection to Bedrock
bedrock_runtime =  helpers.bedrock_runtime_client()

inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=bedrock_runtime,
    model_kwargs=inference_modifier,
)


prompt = '''Human: Archimedes of Syracuse was an Ancient mathematician, physicist, engineer, astronomer, and inventor from the ancient city of Syracuse. Although few details of his life are known, he is regarded as one of the leading scientists in classical antiquity.\n\nWhat was Archimedes? Choose one of the options below.\n\na) astronomer\nb) farmer\nc) sailor\n\nAssistant:'''


with row1_col1:
    with st.form("myform1"):
        prompt_data = st.text_area(
        ":orange[User Prompt:]",
        height = 280,
        value = prompt
        )
        submit = st.form_submit_button("Submit", type="primary")


    if prompt_data and submit:

        response = llm(prompt_data)

        print(response)
        st.write("### Answer")
        st.info(response)
  