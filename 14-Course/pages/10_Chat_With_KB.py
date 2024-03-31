from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain.llms.bedrock import Bedrock
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain

REGION_NAME = "us-east-1"
model_id = "anthropic.claude-v2:1"
knowledge_base_id = "your_knowledge_base_id"

retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=knowledge_base_id,
        region_name=REGION_NAME,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 4
            }
        }
    )

llm = Bedrock(
        model_id=model_id,
        region_name=REGION_NAME,
        model_kwargs={
            "temperature": 0.7, 
            "max_tokens_to_sample": 500
            }
        )

history = ChatMessageHistory()
history.add_user_message("Hello!")
history.add_ai_message("How's going?")

memory = ConversationSummaryBufferMemory(
    llm=llm,
    chat_memory=history, 
    input_key="question",
    memory_key="summary",
    output_key="answer",
)

chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True, 
        get_chat_history=lambda h : h,
        verbose=True
    )