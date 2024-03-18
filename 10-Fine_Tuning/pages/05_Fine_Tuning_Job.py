import time
from datetime import datetime
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock = helpers.bedrock_client()



if "epochCount" not in st.session_state:
    st.session_state.epochCount = 2
if "batchSize" not in st.session_state:
    st.session_state.batchSize = 1
if "learningRate" not in st.session_state:
    st.session_state.learningRate = 0.00003
if "learningRateWarmupSteps" not in st.session_state:
    st.session_state.learningRateWarmupSteps = 5
if "jobName" not in st.session_state:
    st.session_state.jobName = "model-finetune-job"
if "is_job_running" not in st.session_state:
    st.session_state.is_job_running = False


st.title("Create fine-tuning job")
st.markdown("""Amazon Titan text model customization hyperparameters:

- epochs: The number of iterations through the entire training dataset and can take up any integer values in the range of 1-10, with a default value of 5.
- batchSize: The number of samples processed before updating model parametersand can take up any integer values in the range of 1-64, with a default value of 1.
- learningRate: The rate at which model parameters are updated after each batch which can take up a float value betweek 0.0-1.0 with a default value set to 1.00E-5.
- learningRateWarmupSteps: The number of iterations over which the learning rate is gradually increased to the specified rate and can take any integer value between 0-250 with a default value of 5.
""")  

text, code = st.columns([0.6, 0.4])

ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
base_model_id = "amazon.titan-text-lite-v1:0:4k"
customization_job_name = f"model-finetune-job-{ts}"
custom_model_name = f"finetuned-model-{ts}"
customization_role = 'arn:aws:iam::875692608981:role/AmazonBedrockCustomizationRole1'
customization_type = "FINE_TUNING"
bucket_name = 'bedrock-customization-us-east-1-875692608981'
s3_train_uri = 's3://bedrock-customization-us-east-1-875692608981/fine-tuning-datasets/train/train-cnn-5K.jsonl'
s3_validation_uri = 's3://bedrock-customization-us-east-1-875692608981/fine-tuning-datasets/validation/validation-cnn-1K.jsonl'

hyper_parameters = {
        "epochCount": str(st.session_state.epochCount),
        "batchSize": str(st.session_state.batchSize),
        "learningRate": str(st.session_state.learningRate),
        "learningRateWarmupSteps": str(st.session_state.learningRateWarmupSteps)
    }

s3_bucket_config=f's3://{bucket_name}/outputs/output-{custom_model_name}'
# Specify your data path for training, validation(optional) and output
training_data_config = {"s3Uri": s3_train_uri}

validation_data_config = {
        "validators": [{
            "s3Uri": s3_validation_uri
        }]
    }

output_data_config = {"s3Uri": s3_bucket_config}

def set_hyperparameters(epochCount,batchSize,learningRate,learningRateWarmupSteps):
    st.session_state.epochCount = epochCount
    st.session_state.batchSize = batchSize
    st.session_state.learningRate = learningRate
    st.session_state.learningRateWarmupSteps = learningRateWarmupSteps
    

with code:

    with st.container(border=True):
        st.write("Tune the following Hyperparameters")
        epochCount = st.slider("**epochCount**", value=int(hyper_parameters["epochCount"]), min_value=1, max_value=10, step=1)
        batchSize = st.slider("**batchSize**", value=int(hyper_parameters["batchSize"]),min_value=1, max_value=64, step=1)
        learningRate = st.slider("**learningRate**", value=float(hyper_parameters["learningRate"]), min_value=0.0, max_value=0.0001, step=0.00001,format="%.5f")
        learningRateWarmupSteps = st.slider("**learningRateWarmupSteps**", value=int(hyper_parameters["learningRateWarmupSteps"]), min_value=0, max_value=250, step=1)

        col1,col2,col3 = st.columns([0.5,0.3,0.2])
        with col1:
            st.button("Tune Hyperparameters", on_click=set_hyperparameters, args=(epochCount,batchSize,learningRate,learningRateWarmupSteps))
        with col2:
            st.button("Reset", on_click=helpers.set_defaults)

with text:
    with st.form("job"):
        st.write("Submit the following details to create a fine-tuning job")
        customization_job_name = st.text_input("**Job Name**", value = customization_job_name)
        customization_type = st.selectbox("**Customization Type**", ["FINE_TUNING","CONTINUED_PRE_TRAINING"],index=0)
        custom_model_name = st.text_input("**Model Name**", value = custom_model_name)
        customization_role = st.text_input("**Role**", value = customization_role)
        base_model_id = st.selectbox("**Base Model**", helpers.get_models("Amazon"),index=3)
        s3_train_uri  = st.text_input("**Training Data**", value = s3_train_uri)
        s3_validation_uri = st.text_input("**Validation Data**", value = s3_validation_uri)
        s3_bucket_config = st.text_input("**Output Data**", value = s3_bucket_config)
        submit2 = st.form_submit_button("Submit Job", type="primary", disabled=st.session_state.is_disabled)

if submit2:
    #Create the customization job
    training_job_response = bedrock.create_model_customization_job(
        customizationType=customization_type,
        jobName=customization_job_name,
        customModelName=custom_model_name,
        roleArn=customization_role,
        baseModelIdentifier=base_model_id,
        hyperParameters=hyper_parameters,
        trainingDataConfig=training_data_config,
        validationDataConfig=validation_data_config,
        outputDataConfig=output_data_config
    )
    st.session_state.jobName = customization_job_name
    st.session_state.is_job_running = True
    st.success("Job submitted")
    
 
def check_job_status(job_name):

    try:
        with st.spinner("Checking Job Status..."):
            fine_tune_job = bedrock.get_model_customization_job(jobIdentifier=job_name)["status"]
            st.info(f"Job Status: {fine_tune_job}")

    except Exception as e:
        st.warning(f"Error, {str(e)}")
        st.stop()    
        
        
            
with code.container(border=True):
    job_name = st.selectbox("Select Job", helpers.list_jobs('FINE_TUNING'))
    if st.button("Check Job Status"):
        check_job_status(job_name)