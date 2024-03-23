from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, S3FileLoader
from pypdf import PdfReader, PdfWriter
from urllib.request import urlretrieve
import boto3


def txt_loader(file):
    """
    Load a CSV file into a list of documents.
    """
    loader = TextLoader(file)
    documents = loader.load()
    
    return documents

def csv_loader(file):
    """
    Load a CSV file into a list of documents.
    """
    loader = CSVLoader(file)
    documents = loader.load()
    
    return documents

def pdf_loader(file):
    """
    Load a PDF file into a list of documents.
    """
    loader = PyPDFLoader(file)
    document = loader.load_and_split()

    return document

def s3_loader(client,bucket,file):

    loader = S3FileLoader(client, bucket, file )
    document = loader.load()

    return document