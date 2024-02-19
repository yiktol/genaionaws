from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader, PdfWriter
from urllib.request import urlretrieve
import boto3

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

# url= 'https://training.yikyakyuk.com/genai/docs/bedrock_faqs.csv'
# file_path = 'bedrock_faqs.csv'

# urlretrieve(url, file_path)

def csv_loader(file_path):
    """
    Load a CSV file into a list of documents.
    """
    loader = CSVLoader(file_path)
    documents = loader.load()
    
    return documents


pdf_url= 'https://training.yikyakyuk.com/genai/docs/AMZN-2022-Shareholder-Letter.pdf'  
urlretrieve(pdf_url)

def pdf_loader(url):
    """
    Load a PDF file into a list of documents.
    """
    loader = OnlinePDFLoader(pdf_url)
    document = loader.load()

    return document

print(pdf_loader(pdf_url))



def test_loader(file_path):
#Access the content and metadata of each document
    for document in csv_loader(file_path):
        content = document.page_content
        metadata = document.metadata

        print(content)
        print(metadata)
    