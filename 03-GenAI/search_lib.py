from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
import boto3

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = BedrockEmbeddings(client=bedrock_runtime) #create a Titan Embeddings client
    
    loader = CSVLoader(file_path="/home/ubuntu/GENAI/genaionaws/03-GenAI/bedrock_faqs.csv")
    
    documents = loader.load()

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=CharacterTextSplitter(chunk_size=300, chunk_overlap=0),
    )

    index_from_loader = index_creator.from_loaders([loader])
    
    return index_from_loader
    

def get_similarity_search_results(index, question):
    results = index.vectorstore.similarity_search_with_score(question)
    
    flattened_results = [{"content":res[0].page_content, "score":res[1]} for res in results] #flatten results for easier display and handling
    
    return flattened_results
