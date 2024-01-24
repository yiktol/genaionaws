import streamlit as st
from helpers import getmodelId, getmodelparams, set_page_config, bedrock_runtime_client

set_page_config()

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


