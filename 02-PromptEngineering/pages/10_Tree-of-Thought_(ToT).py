import streamlit as st
import utils.helpers as helpers
import uuid

helpers.set_page_config()

questions = [
	{"id": 1, "prompt": """Jennifer purchased 40 cans of milk at the store before meeting her classmate Mark, who was also buying milk. Jennifer bought 6 additional \
cans for every 5 cans Mark bought. If Mark purchased 50 cans, how many cans of milk did Jennifer bring home from the store?""",
	 "context": """Identify and behave as three different experts that are appropriate to answering this question. \
All experts will write down the step and their thinking about the step, then share it with the group. \
Then, all experts will go on to the next step, etc. \n
At each step all experts will score their peers response between 1 and 5, 1 meaning it is highly unlikely, and 5 meaning it is highly likely. \
If any expert is judged to be wrong at any point then they leave. \n
After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus solution or your best guess solution.\n
For clarity, your entire response should be in a markdown table. The question is...
  """},
	{"id": 2, "prompt": """Bob is in the living room.
He walks to the kitchen, carrying a cup.
He puts a ball in the cup and carries the cup to the bedroom.
He turns the cup upside down, then walks to the garden.
He puts the cup down in the garden, then walks to the garage.
Where is the ball?""",
	 "context": """Simulate three brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table. The question is...

  """},
	{"id": 3, "prompt": "",
	 "context": """"""}

]


suffix = 'challenge1'
if suffix not in st.session_state:
	st.session_state[suffix] = {}
if 'height' not in st.session_state[suffix]:
	st.session_state[suffix]['height'] = 100
if 'prompt' not in st.session_state[suffix]:
	st.session_state[suffix]['prompt'] = ""
if 'model' not in st.session_state[suffix]:
	st.session_state[suffix]['model'] = "amazon.titan-tg1-large"
if 'maxTokenCount' not in st.session_state[suffix]:
	st.session_state[suffix]['maxTokenCount'] = 1024
if 'temperature' not in st.session_state[suffix]:
	st.session_state[suffix]['temperature'] = 0.1
if 'topP' not in st.session_state[suffix]:
	st.session_state[suffix]['topP'] = 0.9
if 'topK' not in st.session_state[suffix]:
	st.session_state[suffix]['topK'] = 100

topK = 100

st.subheader("Tree-of-Thought (ToT)")
st.markdown("""Tree-of-thought (ToT) framework involves exploring potential solutions in a manner akin to navigating a tree structure of thoughts, similar to human problem-solving. \
This approach enables the possibility of retracing steps when needed, mirroring the way humans may reassess and adjust their thinking during the problem-solving process. \
In essence, ToT aims to replicate the adaptive and iterative nature of human reasoning through trial and error.""")




text, code = st.columns([0.7, 0.3])


with code:

	with st.container(border=True):
		provider = st.selectbox('provider', helpers.list_providers)
		models = helpers.getmodelIds(provider)
		model = st.selectbox(
			'model', models, index=models.index(helpers.getmodelId(provider)))
	with st.container(border=True):
		params = helpers.tune_parameters(provider)

with text:
	with st.expander("See Diagram"):
		st.image("images/tot.webp", width=500, use_column_width=False)
  
	tab1, tab2 = st.tabs(["ToT-1", "ToT-2"])

	with tab1:
		# st.markdown(questions[0]['context'])
		output = helpers.prompt_box(questions[0]['id'], provider,
							model,
       						context=f"{questions[0]['context']}\n{questions[0]['prompt']}",
             				height=400,
							**params)
		
		if output:
			st.write("### Answer")
			st.info(output)

	with tab2:
		# st.markdown(questions[1]['context'])
		output = helpers.prompt_box(questions[1]['id'], provider,
							model,
       						context=f"{questions[1]['context']}\n{questions[1]['prompt']}",
							height=350,
							**params)
		
		if output:
			st.write("### Answer")
			st.info(output)
