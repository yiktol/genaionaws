import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import utils.loader as docloader
from urllib.request import urlretrieve
import boto3

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)

st.header("Document loaders")
st.markdown("""Use document loaders to load data from a source as Document's. \
A Document is a piece of text and associated metadata. For example, there are document loaders for loading a simple \
.txt file, for loading the text contents of any web page, or even for loading a transcript of a YouTube video.""")

st.subheader(":orange[TXT]")
st.markdown("The simplest loader reads in a file as text and places it all into one document. \
[Sample TXT file](https://training.yikyakyuk.com/genai/docs/alice.txt)")

expander = st.expander("See code")
expander.code("""from langchain_community.document_loaders import TextLoader

loader = TextLoader("./index.txt")
loader.load()""",language="python")

def txt_loader(file):
    """
    Load a CSV file into a list of documents.
    """
    loader = TextLoader(file)
    documents = loader.load()
    
    return documents

with st.form("form3"):
    uploaded_file = st.file_uploader("Choose a TXT file:", type=['txt'], accept_multiple_files=False)
    submit = st.form_submit_button("Load TXT Data",type="primary")

    if submit:
        temp_file = "./temp.txt"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        documents = docloader.txt_loader(temp_file)
        st.write(documents)

st.subheader(":orange[CSV]")
st.markdown("A comma-separated values (CSV) file is a delimited text file that uses a comma to separate values. Each line of the file is a data record.\
Each record consists of one or more fields, separated by commas. [Sample CSV file](https://training.yikyakyuk.com/genai/docs/bedrock_faqs.csv)")

expander = st.expander("See code")
expander.code("""from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='bedrock_faqs.csv')
data = loader.load()

print(data)""",language="python")


def csv_loader(file):
    """
    Load a CSV file into a list of documents.
    """
    loader = CSVLoader(file)
    documents = loader.load()
    
    return documents


with st.form("form1"):
    uploaded_file = st.file_uploader("Choose a CSV file:", type=['csv'],accept_multiple_files=False)
    submit = st.form_submit_button("Load CSV Data",type="primary")

    if submit:
        temp_file = "./temp.csv"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        documents = docloader.csv_loader(temp_file)
        st.write(documents)


st.subheader(":orange[PDF]")
st.markdown("Portable Document Format (PDF), standardized as ISO 32000, is a file format developed by Adobe in 1992 to present documents, \
including text formatting and images, in a manner independent of application software, hardware, and operating systems. \
[Sample PDF file](https://training.yikyakyuk.com/genai/docs/AMZN-2022-Shareholder-Letter.pdf)")

expander = st.expander("See code")
expander.code("""from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()""",language="python")


def pdf_loader(file):
    """
    Load a PDF file into a list of documents.
    """
    loader = PyPDFLoader(file)
    document = loader.load_and_split()

    return document

with st.form("form2"):
    uploaded_file = st.file_uploader("Choose a PDF file:",type=['pdf'])
    submit = st.form_submit_button("Load PDF Data",type="primary")

    if submit and uploaded_file:

        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        documents = docloader.pdf_loader(temp_file)
        st.write(documents)

