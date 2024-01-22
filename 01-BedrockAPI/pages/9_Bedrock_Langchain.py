import boto3
from langchain.llms.bedrock import Bedrock
import streamlit as st


#Create the connection to Bedrock
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

st.set_page_config( 
    page_title="Langchain",  
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

text, code = st.columns(2)

with text:
    st.title("Langchain")
    st.write("LangChain is a framework for developing applications powered by language models.")

    inference_modifier = {
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"], }

    textgen_llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock_runtime,
        model_kwargs=inference_modifier,
    )

    with st.form("myform"):
        prompt_data = st.text_area(
            "Ask something:",
            height = 150,
            placeholder="Write me an invitaion letter for my wedding.",
            value = """Human: Write an email from Bob, Customer Service Manager, to the customer "John Doe" that provided negative feedback on the service provided by our customer support engineer. \n\nAssistant:"""
            )
        submit = st.form_submit_button("Submit")

    if prompt_data and submit:

        response = textgen_llm(prompt_data)

        print(response)
        st.write("### Answer")
        st.write(response)
  

with code:

    code = '''
    import boto3
    from langchain.llms.bedrock import Bedrock
    from langchain.prompts import PromptTemplate

    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1', 
    )

    template = 'Human: {task}\\n\\nAssistant:'

    inference_modifier = {
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": [\"\\n\\nHuman\"],
    }

    def call_llm():

        llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock_runtime,
        model_kwargs=inference_modifier
        )

        prompt = PromptTemplate(input_variables=["task"], template=template)
        prompt_query = prompt.format(
                task="Write an email from Bob, Customer Service Manager, to the customer \"John Doe\" \
                      that provided negative feedback on the service provided by our customer support engineer."
                )
        response = llm(prompt_query)
        
        return response

    print(call_llm())

    '''

    st.code(code,language="python")
