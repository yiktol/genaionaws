import json
import streamlit as st
import utils.helpers as helpers
from bert_score import score
import threading


helpers.set_page_config()
bedrock = helpers.bedrock_client()
bedrock_runtime = helpers.bedrock_runtime_client()
region = 'us-east-1'


test_file_path = 'data/test-cnn-10.jsonl'
with open(test_file_path) as f:
    lines = f.read().splitlines()

test_prompt = json.loads(lines[3])['prompt']
reference_summary = json.loads(lines[3])['completion']


def get_provisioned_model_id():
    provisioned_model_throughput_id = bedrock.list_provisioned_model_throughputs()

    status = provisioned_model_throughput_id['provisionedModelSummaries'][0]['status']

    if status in ['Creating', 'Updating', 'InService']:
        id = provisioned_model_throughput_id['provisionedModelSummaries'][0]['provisionedModelArn']
    else:
        id = None

    return id


st.title("Model Evaluation")
st.markdown("""In this section, we will use BertScore metrics to evaluate the performance of the fine-tuned model as compared to base model to check if fine-tuning has improved the results.

_BERTScore_: calculates the similarity between a summary and reference texts based on the outputs of BERT (Bidirectional Encoder Representations from Transformers), a powerful language model.
""")

with st.expander(":orange[See Sample Test Prompt]"):
    st.subheader("Test Prompt")
    st.write(test_prompt)
    st.subheader("Reference Summary")
    st.write(reference_summary)


prompt = f"""{test_prompt}"""

base_model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1'

accept = 'application/json'
contentType = 'application/json'

with st.container(border=True):
    prompt_data = st.text_area(":orange[Prompt]", value=prompt, height=300)
    submit = st.button("Evaluate Models", type="primary")


body = json.dumps(
    {
        "inputText": prompt_data,
        "textGenerationConfig": {
            "maxTokenCount": 2048,
            "stopSequences": ['User:'],
            "temperature": 0,
            "topP": 0.9
        }
    }
)


def invoke_ftm():
    global fine_tuned_response_body
    with st.spinner("Invoking the Fine Tuned Models..."):

        if helpers.get_provisioned_model_id() is None:
            st.write("Fine tuned model response: ")
            st.error("Please Submit the Provisioned Throughput Request")
            # st.stop()
        else:
            fine_tuned_response = bedrock_runtime.invoke_model(body=body,
                                                               modelId=get_provisioned_model_id(),
                                                               accept=accept,
                                                               contentType=contentType)
            fine_tuned_response_body = json.loads(
                fine_tuned_response.get('body').read())
            st.write("Fine tuned model response: ")
            st.success(
                fine_tuned_response_body["results"][0]["outputText"] + '\n')


def invoke_bm():
    global base_model_response_body
    with st.spinner("Invoking the Base Models..."):
        base_model_response = bedrock_runtime.invoke_model(body=body,
                                                           modelId=base_model_arn,
                                                           accept=accept,
                                                           contentType=contentType)
        base_model_response_body = json.loads(
            base_model_response.get('body').read())
        st.write("Base model response:")
        st.success(base_model_response_body["results"][0]["outputText"] + '\n')


if submit:

    col1, col2 = st.columns(2)

    with col1:
        t1 = threading.Thread(target=invoke_ftm())
        t1.start()
        t1.join()
    with col2:
        t2 = threading.Thread(target=invoke_bm())
        t2.start()
        t2.join()

    base_model_generated_response = [
        base_model_response_body["results"][0]["outputText"]]
    fine_tuned_generated_response = [
        fine_tuned_response_body["results"][0]["outputText"]]

    with st.spinner("Calculating Performance..."):
        reference_summary = [reference_summary]
        fine_tuned_model_P, fine_tuned_R, fine_tuned_F1 = score(
            fine_tuned_generated_response, reference_summary, lang="en")
        base_model_P, base_model_R, base_model_F1 = score(
            base_model_generated_response, reference_summary, lang="en")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("F1 score: Base Model ", base_model_F1)
            st.metric("F1 score: Fine-Tuned Model", fine_tuned_F1,
                      delta=fine_tuned_F1.item() - base_model_F1.item())
        with col2:
            st.metric("Precision: Base Model ", base_model_P)
            st.metric("Precision: Fine-Tuned Model", fine_tuned_model_P,
                      delta=fine_tuned_model_P.item() - base_model_P.item())
        with col3:
            st.metric("Recall: Base Model ", base_model_R)
            st.metric("Recall: Fine-Tuned Model", fine_tuned_R,
                      delta=fine_tuned_R.item() - base_model_R.item())
