import streamlit as st
from utils import set_page_config

set_page_config()

for key in st.session_state.keys():
    del st.session_state[key]

st.title("Amazon Bedrock API")

intro = '''

This section describes how to set up your environment to make Amazon Bedrock API calls and provides examples of common use-cases. You can access the Amazon Bedrock API using the AWS Command Line Interface (AWS CLI), an AWS SDK, or a SageMaker Notebook.

Before you can access Amazon Bedrock APIs, you need to request access to the foundation models that you plan to use.


### Amazon Bedrock endpoints

To connect programmatically to an AWS service, you use an endpoint. Refer to the Amazon Bedrock endpoints and quotas chapter in the AWS General Reference for information about the endpoints that you can use for Amazon Bedrock.

Amazon Bedrock provides the following service endpoints.

- :orange[bedrock]:  Contains control plane APIs for managing, training, and deploying models. 
- :orange[bedrock-runtime]: Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock.
- :orange[bedrock-agent]: Contains control plane APIs for creating and managing agents and knowledge bases.
- :orange[bedrock-agent-runtime]: Contains control plane APIs for managing, training, and deploying models.
'''

st.markdown(intro)

st.subheader("invoke_model")
st.image("images/invoke_model.png")
st.markdown("""Invokes the specified Bedrock model to run inference using the input provided in the request body. You use InvokeModel to run inference for text models, image models, and embedding models.

Parameters:
- body (bytes or seekable file-like object) - Input data in the format specified in the content-type request header. To see the format and content of this field for different models, refer to Inference parameters.
- contentType (string) - The MIME type of the input data in the request. The default value is application/json.
- accept (string) - The desired MIME type of the inference body in the response. The default value is application/json.
- modelId (string) - Identifier of the model.

            """)

st.subheader("invoke_model_with_response_stream")
st.image("images/invoke_model_streaming.png")
st.markdown("""Invoke the specified Bedrock model to run inference using the input provided. Return the response in a stream.""")

st.subheader("Invoke Model API, Model Parameters")
st.image("images/model_parameters.png")
