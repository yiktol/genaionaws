# Python program to generate word vectors using Word2Vec
import streamlit as st
# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import numpy as np
import nltk
from helpers import set_page_config

set_page_config()
nltk.download('punkt')
warnings.filterwarnings(action='ignore')

st.title("Vertorizer using using Word2Vec")

vector_size = st.slider(":orange[Dimensions]",min_value=1,max_value=10,value=1,step = 1)

with st.form("myform"):
    f = st.text_input("Enter a prompt to vectorize", 
                      placeholder="Hello World",
                      value="Hello World")
    submitted = st.form_submit_button("Vectorize")

data = []

if f:
# iterate through each sentence in the file
	for i in sent_tokenize(f):
		temp = []

		# tokenize the sentence into words
		for j in word_tokenize(i):
			temp.append(j.lower())
			# print(temp)

		data.append(temp)
	

	# Create CBOW model
	model1 = gensim.models.Word2Vec(data, min_count=1,
									vector_size=vector_size,)
	# print(model1.wv.vectors)

 
	x = np.array(model1.wv.vectors)
	st.write(x)
