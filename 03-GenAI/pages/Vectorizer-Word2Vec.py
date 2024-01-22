# Python program to generate word vectors using Word2Vec
import streamlit as st
# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import numpy as np

warnings.filterwarnings(action='ignore')

with st.sidebar:
    with st.form(key ='Form1'):
        "Parameters:"
        vector_size =st.number_input('vector_size',min_value = 1, max_value = 10, value = 1, step = 1)
        submitted1 = st.form_submit_button(label = 'Set Parameters') 

st.title("Vertorizer using using Word2Vec")

with st.form("myform"):
    f = st.text_input("Enter a prompt to vetorize", 
                      placeholder="Hello World",
                      value="Hello World")
    submitted = st.form_submit_button("Vectorize")

data = []

if f and submitted:
# iterate through each sentence in the file
	for i in sent_tokenize(f):
		temp = []

		# tokenize the sentence into words
		for j in word_tokenize(i):
			temp.append(j.lower())
			print(temp)

		data.append(temp)
	

	# Create CBOW model
	model1 = gensim.models.Word2Vec(data, min_count=1,
									vector_size=vector_size,)
	print(model1.wv.vectors)

 
	x = np.array(model1.wv.vectors)
	st.write(x)
