import os
import boto3
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrock

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

def get_llm():
        
    model_kwargs = { #AI21
        "maxTokenCount": 1024, 
        "temperature": 1
        # "top_p": 0.5
    }
    
    llm = ChatBedrock(
        client=bedrock,
        model_id="amazon.titan-text-premier-v1:0", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm


def get_memory(): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    #this allows us to maintain the "big picture" of a long-running conversation
    llm = get_llm()
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024) #Maintains a summary of previous messages
    
    return memory


def get_chat_response(input_text, memory): #chat client function
    
    llm = get_llm()
    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, #using the Bedrock LLM
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    
    chat_response = conversation_with_summary.predict(input=input_text) #pass the user message and summary to the model
    
    return chat_response
