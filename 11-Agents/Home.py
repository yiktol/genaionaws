import streamlit as st #all streamlit commands will be available through the "st" alias
from utils.helpers import set_page_config


set_page_config()

st.title("Agents for Bedrock")

st.markdown("""Agents for Amazon Bedrock offers you the ability to build and configure autonomous agents in your application. \
An agent helps your end-users complete actions based on organization data and user input. \
Agents orchestrate interactions between foundation models (FMs), data sources, software applications, and user conversations. \
In addition, agents automatically call APIs to take actions and invoke knowledge bases to supplement information for these actions. \
Developers can save weeks of development effort by integrating agents to accelerate the delivery of generative artificial intelligence (generative AI) applications.

With agents, you can automate tasks for your customers and answer questions for them. For example, you can create an agent that helps customers process insurance claims or an agent that helps customers make travel reservations. \
You don't have to provision capacity, manage infrastructure, or write custom code. Amazon Bedrock manages prompt engineering, memory, monitoring, encryption, user permissions, and API invocation.

Agents perform the following tasks:
- Extend foundation models to understand user requests and break down the tasks that the agent must perform into smaller steps.
- Collect additional information from a user through natural conversation.
- Take actions to fulfill a customer's request by making API calls to your company systems.
- Augment performance and accuracy by querying data sources.
            """, unsafe_allow_html=True)
