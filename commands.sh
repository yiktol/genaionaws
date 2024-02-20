
cd /home/ubuntu/environment
pip install -r genaionaws/requirements.txt
streamlit run genaionaws/01-BedrockAPI/Home.py --server.port 8080 &
streamlit run genaionaws/02-PromptEngineering/Home.py --server.port 8081 &
streamlit run genaionaws/03-GenAI/Home.py --server.port 8082 &
cd /home/ubuntu/environment/genaionaws/05-Chatbot
streamlit run Home.py --server.port 8083 &
cd /home/ubuntu/environment/genaionaws/04-Embedding
streamlit run Home.py --server.port 8084 &
cd /home/ubuntu/environment/genaionaws/06-UseCases
streamlit run Home.py --server.port 8085 &
cd /home/ubuntu/environment/genaionaws/07-LangChain
streamlit run Home.py --server.port 8086 &

chown ubuntu:root /home/ubuntu/environment -R
git config --global --add safe.directory /home/ubuntu/environment/genaionaws