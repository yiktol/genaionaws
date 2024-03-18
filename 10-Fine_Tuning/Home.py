import streamlit as st


# Delete all the items in Session state
for key in st.session_state.keys():
    del st.session_state[key]

if "is_disabled" not in st.session_state:
    st.session_state.is_disabled = True

st.set_page_config(
    page_title="Fine Tuning",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Model Fine Tuning")
st.markdown("""Model customization is the process of providing training data to a model in order to improve its performance for specific use-cases. \
You can customize Amazon Bedrock foundation models in order to improve their performance and create a better customer experience. \
Amazon Bedrock currently provides the following customization methods. 

- _Continued Pre-training_ - Provide unlabeled data to pre-train a foundation model by familiarizing it with certain types of inputs. \
You can provide data from specific topics in order to expose a model to those areas. \
The Continued Pre-training process will tweak the model parameters to accommodate the input data and improve its domain knowledge.

- _Fine-tuning_ - Provide labeled data in order to train a model to improve performance on specific tasks. \
By providing a training dataset of labeled examples, the model learns to associate what types of outputs should be generated for certain types of inputs. \
The model parameters are adjusted in the process and the model's performance is improved for the tasks represented by the training dataset.

""") 

st.image("images/finetune.webp")