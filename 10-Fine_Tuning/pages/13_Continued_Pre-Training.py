from datetime import datetime
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock = helpers.bedrock_client()

if "epochCount" not in st.session_state:
    st.session_state.epochCount = 1
if "batchSize" not in st.session_state:
    st.session_state.batchSize = 1
if "learningRate" not in st.session_state:
    st.session_state.learningRate = 0.00005
if "learningRateWarmupSteps" not in st.session_state:
    st.session_state.learningRateWarmupSteps = 5
if "jobName" not in st.session_state:
    st.session_state.jobName = "model-cpt-job"
if "is_job_running" not in st.session_state:
    st.session_state.is_job_running = False

st.title("Continued Pre-training Foundation Models")
st.markdown("""Provide unlabeled data to pre-train a foundation model by familiarizing it with certain types of inputs. You can provide data from specific topics in order to expose a model to those areas. \
The Continued Pre-training process will tweak the model parameters to accommodate the input data and improve its domain knowledge. \
For example, you can train a model with private data, such as business documents, that are not publically available for training large language models. \
Additionally, you can continue to improve the model by retraining the model with more unlabeled data as it becomes available.
""")


with st.expander(":orange[Preparing a Continued Pre-training dataset]"):
    st.markdown(
        """To carry out Continued Pre-training on a text-to-text model, prepare a training and optional validation dataset by creating a JSONL file with multiple JSON lines. Because Continued Pre-training involves unlabeled data, each JSON line is a sample containing only an input field. Use 6 characters per token as an approximation for the number of tokens. The format is as follows.""")
    st.code("""{"input": "<input text>"}

{"input": "<input text>"}

{"input": "<input text>"}""", language='json')
    st.markdown(
        """The following is an example item that could be in the training data:""")
    st.code("""{"input": "AWS stands for Amazon Web Services"}""")

text, code = st.columns([0.6, 0.4])

ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
base_model_id = "amazon.titan-text-express-v1:0:8k"
customization_job_name = f"cpt-job-titan-express-{ts}"
custom_model_name = f"cpt-model-titan-express-{ts}"
customization_role = 'arn:aws:iam::875692608981:role/AmazonBedrockCustomizationRole1'
customization_type = "CONTINUED_PRE_TRAINING"
bucket_name = 'bedrock-customization-us-east-1-875692608981'
s3_train_uri = 's3://bedrock-customization-us-east-1-875692608981/continued-pretraining/train/aws-cli-dataset.jsonl'
# s3_validation_uri = 's3://bedrock-customization-us-east-1-875692608981/fine-tuning-datasets/validation/validation-cnn-1K.jsonl'

hyper_parameters = {
    "epochCount": str(st.session_state.epochCount),
    "batchSize": str(st.session_state.batchSize),
    "learningRate": str(st.session_state.learningRate),
    "learningRateWarmupSteps": str(st.session_state.learningRateWarmupSteps)
}

s3_bucket_config = f's3://{bucket_name}/outputs/output-{custom_model_name}'
# Specify your data path for training, validation(optional) and output
training_data_config = {"s3Uri": s3_train_uri}

# validation_data_config = {
#         "validators": [{
#             "s3Uri": s3_validation_uri
#         }]
#     }

output_data_config = {"s3Uri": s3_bucket_config}


def set_hyperparameters(epochCount, batchSize, learningRate, learningRateWarmupSteps):
    st.session_state.epochCount = epochCount
    st.session_state.batchSize = batchSize
    st.session_state.learningRate = learningRate
    st.session_state.learningRateWarmupSteps = learningRateWarmupSteps


def get_models(provider):

   # Let's see all available Amazon Models
    available_models = bedrock.list_foundation_models()

    models = []

    for each_model in available_models['modelSummaries']:
        if provider in each_model['providerName']:
            models.append(each_model['modelId'])
    models.pop(0)

    return models


with code:

    with st.container(border=True):
        st.write("Tune the following Hyperparameters")
        epochCount = st.slider(
            "**epochCount**", value=st.session_state.epochCount, min_value=1, max_value=10, step=1)
        batchSize = st.slider(
            "**batchSize**", value=st.session_state.batchSize, min_value=1, max_value=10, step=1)
        learningRate = st.slider("**learningRate**", value=st.session_state.learningRate,
                                 min_value=0.00000, max_value=0.0001, step=0.00001, format="%.5f")
        learningRateWarmupSteps = st.slider(
            "**learningRateWarmupSteps**", value=st.session_state.learningRateWarmupSteps, min_value=0, max_value=10, step=1)

        col1, col2, col3 = st.columns([0.5, 0.4, 0.1])
        with col1:
            st.button("Tune Hyperparameters", on_click=set_hyperparameters, args=(
                epochCount, batchSize, learningRate, learningRateWarmupSteps))
        with col2:
            st.button("Reset", on_click=helpers.set_defaults)

with text:
    with st.form("job"):
        st.write(
            "Submit the following details to create a continued pre-training job")
        customization_job_name = st.text_input(
            "**Job Name**", value=customization_job_name)
        customization_type = st.selectbox(
            "**Customization Type**", ["FINE_TUNING", "CONTINUED_PRE_TRAINING"], index=1)
        custom_model_name = st.text_input(
            "**Model Name**", value=custom_model_name)
        customization_role = st.text_input(
            "**Role**", value=customization_role)
        base_model_id = st.selectbox(
            "**Base Model**", get_models("Amazon"), index=5)
        s3_train_uri = st.text_input("**Training Data**", value=s3_train_uri)
        # s3_validation_uri = st.text_input("**Validation Data**", value = s3_validation_uri)
        s3_bucket_config = st.text_input(
            "**Output Data**", value=s3_bucket_config)
        submit2 = st.form_submit_button("Submit Job", type="primary", disabled=st.session_state.is_disabled)

if submit2:
    # Create the customization job
    training_job_response = bedrock.create_model_customization_job(
        customizationType=customization_type,
        jobName=customization_job_name,
        customModelName=custom_model_name,
        roleArn=customization_role,
        baseModelIdentifier=base_model_id,
        hyperParameters=hyper_parameters,
        trainingDataConfig=training_data_config,
        # validationDataConfig=validation_data_config,
        outputDataConfig=output_data_config
    )
    st.session_state.jobName = customization_job_name
    st.session_state.is_job_running = True
    st.success("Job submitted")


def check_job_status(job_name):

    try:
        with st.spinner("Checking Job Status..."):
            cpt_job = bedrock.get_model_customization_job(
                jobIdentifier=job_name)["status"]
            st.info(f"Job Status: {cpt_job}")

    except Exception as e:
        st.warning(f"Error, {str(e)}")
        st.stop()


with code.container(border=True):
    job_name = st.selectbox(
        "Select Job", helpers.list_jobs('CONTINUED_PRE_TRAINING'))
    if st.button("Check Job Status"):
        check_job_status(job_name)
