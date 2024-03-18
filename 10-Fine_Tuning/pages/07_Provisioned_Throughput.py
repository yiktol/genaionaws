import time
import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()
bedrock = helpers.bedrock_client()


if "custom_model_name" not in st.session_state:
    st.session_state["custom_model_name"] = "finetuned-model-2024-03-14-00-47-13"
if "is_request_submitted" not in st.session_state:
    st.session_state["is_request_submitted"] = False
if "provisioned_model_id" not in st.session_state:
    st.session_state["provisioned_model_id"] = st.empty()


st.title("Provisioned Throughput")
st.markdown("""When you configure Provisioned Throughput for a model, you receive a level of throughput at a fixed cost. \
You can use Provisioned Throughput with Amazon and third-party base models, and with customized models. \
Provisioned Throughput pricing varies depending on the model that you use and the level of commitment you choose. You receive a discounted rate when you commit to a longer period of time.

You specify Provisioned Throughput in Model Units (MU). A model unit delivers a specific throughput level for the specified model. \
The throughput level of a MU for a given Text model specifies the following:
- _The total number of input tokens per minute_ - The number of input tokens that an MU can process across all requests within a span of one minute.
- _The total number of output tokens per minute_ - The number of output tokens that an MU can generate across all requests within a span of one minute.

""")


custom_models = []
custom_models_names = []
for item in bedrock.list_custom_models()['modelSummaries']:
    model = {}
    model['Name'] = item['modelName']
    model['CustomizationType'] = item['customizationType']
    model['BaseModelName'] = item['baseModelArn'].split('/')[-1]
    custom_models_names.append(model['Name'])
    custom_models.append(model)


def delete_provisioned_throughput(provisioned_model_id):
    bedrock.delete_provisioned_model_throughput(
        provisionedModelId=provisioned_model_id)
    with st.spinner("Deleting Provisioned Throughput.."):
        st.success("Provisioned Throughput Deleted")
        st.stop()


def check_status(provisioned_model_id):

    try:
        status_provisioning = bedrock.get_provisioned_model_throughput(
            provisionedModelId=provisioned_model_id)['status']

        if status_provisioning == 'Creating':
            st.info(status_provisioning)
        elif status_provisioning == 'InService':
            st.success("Provisioned Throughput Created")
            st.stop()
        elif status_provisioning == 'Failed':
            st.error("Provisioned Throughput Failed")
            st.stop()
        elif status_provisioning == 'Updating':
            st.warning("Provisioned Throughput Updating")
            st.stop()
    except Exception as e:
        st.error(f"Error", str(e))
        st.stop()


def get_provisioned_model_id():

    ids = []
    provisioned_model_throughput_id = bedrock.list_provisioned_model_throughputs()

    for id in provisioned_model_throughput_id['provisionedModelSummaries']:
        ids.append(id['provisionedModelArn'])

    return ids


text, code = st.columns([0.5, 0.5])

with text:
    st.subheader("Request Provisioned Throughput")
    with st.form("model_form"):
        model_name = st.selectbox(
            ":orange[Select a Custom Model]", custom_models_names, index=1, key="model_id")
        model_units = st.number_input(":orange[Model Units]", min_value=1, max_value=2, value=1, key="model_units",
                                      help="Model Units are the number of MU that you want to provision for a model.")
        submit = st.form_submit_button(
            "Create Provisioned Throughput", type="primary", disabled=st.session_state.is_disabled)

    if submit and not st.session_state["is_request_submitted"]:
        with st.spinner("Creating Provisioned Throughput.."):
            model_id = bedrock.get_custom_model(
                modelIdentifier=model_name)['modelArn']
            try:
                provisioned_model_id = bedrock.create_provisioned_model_throughput(
                    modelUnits=model_units,
                    provisionedModelName=f'custom-model-{model_name}',
                    modelId=model_id)['provisionedModelArn']
                st.session_state["is_request_submitted"] = True
                st.session_state["provisioned_model_id"] = provisioned_model_id
                st.success("Provisioned Throughput Request Submitted.")
            except Exception as e:
                st.error(f"Error", str(e))
                st.stop()
    elif submit and st.session_state["is_request_submitted"]:
        st.info("Provisioned Throughput Request Submitted.")


with code:
    st.subheader("Fine Tuned Models")
    # print(custom_models)
    with st.container(border=True):
        st.dataframe(custom_models)

    list_provisioned_model_throughputs = bedrock.list_provisioned_model_throughputs(
        statusEquals='InService')
    provisioned_model_throughputs = []
    for item in list_provisioned_model_throughputs['provisionedModelSummaries']:
        provisioned_model_throughputs.append(item['provisionedModelArn'])
    with st.container(border=True):

        provisionedModelId = st.selectbox(
            ":orange[Provisioned Throughputs]", get_provisioned_model_id())

        col1, col2, col3 = st.columns([0.5, 0.4, 0.1])
        with col1:
            # provisionedModelId1 = st.selectbox("Provisioned Throughputs", provisioned_model_throughputs, key="show_provisioned_throughputs")
            st.button("Delete Provisioned Throughput",
                      on_click=delete_provisioned_throughput, args=(provisionedModelId,))

        with col2:

            check_status_btn = st.button("Check Status")
    if check_status_btn and not st.session_state["is_request_submitted"]:
        st.warning("Please Submit the Provisioned Throughput Request")
        st.stop()
    elif check_status_btn and st.session_state["is_request_submitted"]:
        with st.spinner("Checking Status of the Provisioned Throughput.."):
            check_status(provisionedModelId)
