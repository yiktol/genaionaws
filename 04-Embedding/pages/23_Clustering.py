import json
import boto3
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from helpers import bedrock_runtime_client, set_page_config, classify, search, get_embedding

set_page_config()
bedrock = bedrock_runtime_client()

text, code = st.columns(2)

with text:

    # st.title("Extract Action Items")
    st.header("Clustering")
    st.markdown("""When we look at the following name list, we know some of them are physicists and some of them are musicians. However, if we randomly pick some names from two other fields that we are not familiar with, how can we correctly allocate those names into two groups without knowing what the groups are?\n
_names = ['Albert Einstein', 'Bob Dylan', 'Elvis Presle 'Isaac Newton', 'Michael Jackson', 'Niels Bohr',
'Taylor Swift', 'Hank Williams', 'Werner Heis,rg', \ 'Stevie Wonder', 'Marie Curie', 'Ernest Rutherford']_\n
Clustering is a technique to identify hidden groupings in data, with K-means clustering being a commonly used clustering algorithm. Here “K” refers to the number of groups, which is a hyper-parameter determined by a human. Once the number of groups are determined, documents close to each other are said to belong to the same group.\n
Here is the sample code to place the above-mentioned names into two groups:""")

    with st.form("myform"):
        names = st.multiselect(":orange[Select Names:]", 
                            ['Albert Einstein', 'Bob Dylan', 'Elvis Presley', 
         'Isaac Newton', 'Michael Jackson', 'Niels Bohr', 
         'Taylor Swift', 'Hank Williams', 'Werner Heisenberg', 
         'Stevie Wonder', 'Marie Curie', 'Ernest Rutherford'],['Albert Einstein', 'Bob Dylan', 'Elvis Presley', 
         'Isaac Newton', 'Michael Jackson', 'Niels Bohr', 
         'Taylor Swift', 'Hank Williams', 'Werner Heisenberg', 
         'Stevie Wonder', 'Marie Curie', 'Ernest Rutherford']),
        submit = st.form_submit_button("Find Cluter",type="primary")

    if submit:
        embeddings = []
        for name in names[0]:
            embeddings.append(get_embedding(bedrock, name))
            print(name)
        # clustering
        df = pd.DataFrame(data={'names': names[0], 'embeddings': embeddings})
        matrix = np.vstack(df.embeddings.values)
        n_clusters = 2
        kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
        kmeans.fit(matrix)
        df['cluster'] = kmeans.labels_
        #print(df['cluster'] )
        # result
        #print(df[['cluster', 'names']])
        st.table(df[['cluster', 'names']])

        # Reduce number of dimensions from 1536 to 2
        tsne = TSNE(random_state=0, n_iter=1000, perplexity=6)
        tsne_results = tsne.fit_transform(np.array(df['embeddings'].to_list(), dtype=np.float32))
        # Add the results to dataframe as a new column
        df['tsne1'] = tsne_results[:, 0]
        df['tsne2'] = tsne_results[:, 1]

        # Plot the data and annotate the result
        fig, ax = plt.subplots()
        ax.set_title('Embeddings')
        sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='cluster', ax=ax)
        for idx, row in df.iterrows():
            ax.text(row['tsne1'], row['tsne2'], row['names'], fontsize=6.5, horizontalalignment='center')

        # plt.show()
        st.pyplot(fig)


with code:

    code_data = """import json
import boto3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_embedding(bedrock, text):
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    input = {
            'inputText': text
        }
    body=json.dumps(input)
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept,contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body['embedding']
    return embedding

# main function
region_name = 'us-east-1'
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name
)
names = ['Albert Einstein', 'Bob Dylan', 'Elvis Presley', 
         'Isaac Newton', 'Michael Jackson', 'Niels Bohr', 
         'Taylor Swift', 'Hank Williams', 'Werner Heisenberg', 
         'Stevie Wonder', 'Marie Curie', 'Ernest Rutherford']
embeddings = []
for name in names:
    embeddings.append(get_embedding(bedrock, name))
# clustering
df = pd.DataFrame(data={'names': names, 'embeddings': embeddings})
matrix = np.vstack(df.embeddings.values)
n_clusters = 2
kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)
df['cluster'] = kmeans.labels_
# result
print(df[['cluster', 'names']])

# Reduce number of dimensions from 1536 to 2
tsne = TSNE(random_state=0, n_iter=1000, perplexity=6)
tsne_results = tsne.fit_transform(np.array(df['embeddings'].to_list(), dtype=np.float32))
# Add the results to dataframe as a new column
df['tsne1'] = tsne_results[:, 0]
df['tsne2'] = tsne_results[:, 1]

# Plot the data and annotate the result
fig, ax = plt.subplots()
ax.set_title('Embeddings')
sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='cluster', ax=ax)
for idx, row in df.iterrows():
    ax.text(row['tsne1'], row['tsne2'], row['names'], fontsize=6.5, horizontalalignment='center')

plt.show()
        """

    st.code(code_data, language="python")



            

