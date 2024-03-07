pip install -r /home/ubuntu/environment/genaionaws/requirements.txt

cd /home/ubuntu/environment/genaionaws/01-BedrockAPI
streamlit run Home.py --server.port 8080 &

cd /home/ubuntu/environment/genaionaws/02-PromptEngineering
streamlit run Home.py --server.port 8081 &

cd /home/ubuntu/environment/genaionaws/03-GenAI
streamlit run Home.py --server.port 8082 &

cd /home/ubuntu/environment/genaionaws/05-Chatbot
streamlit run Home.py --server.port 8083 &

cd /home/ubuntu/environment/genaionaws/04-Embedding
streamlit run Home.py --server.port 8084 &

cd /home/ubuntu/environment/genaionaws/06-UseCases
streamlit run Home.py --server.port 8085 &

cd /home/ubuntu/environment/genaionaws/07-LangChain
streamlit run Home.py --server.port 8086 &

cd /home/ubuntu/environment/genaionaws/08-Images
streamlit run Home.py --server.port 8087 &

cd /home/ubuntu/environment/genaionaws/09-Antropic
streamlit run Home.py --server.port 8088 &

chown ubuntu:root /home/ubuntu/environment -R

git config --global --add safe.directory /home/ubuntu/environment/genaionaws