
import streamlit as st
from helpers import set_page_config

set_page_config()

st.title("Cloudwatch Dashboard")
st.write("You can monitor Amazon Bedrock using Amazon CloudWatch, which collects raw data and processes it into readable, near real-time metrics. You can graph the metrics using the CloudWatch console. You can also set alarms that watch for certain thresholds, and send notifications or take actions when values exceed those thresholds.")

st.write(
        f'<iframe src="https://cloudwatch.amazonaws.com/dashboard.html?dashboard=AmazonBedrockDashboard&context=eyJSIjoidXMtZWFzdC0xIiwiRCI6ImN3LWRiLTg3NTY5MjYwODk4MSIsIlUiOiJ1cy1lYXN0LTFfU0NUNmVEM1h0IiwiQyI6IjVxMm9xdjA2M3V2MHAydWhzdWZuMXVrZ2pzIiwiSSI6InVzLWVhc3QtMTpjNzhhZGE0OC0zOGEwLTRjMmItODJhNC00ZTNmYmU5NjIwZjIiLCJNIjoiUHVibGljIn0=" width="1000" height="1000"></iframe>',
        unsafe_allow_html=True,
    )