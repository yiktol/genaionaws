import streamlit as st
import utils.helpers as helpers
import uuid

helpers.set_page_config()

questions = [
	{"id": 1, "prompt": "When I was 6 my sister was half my age. \nNow I'm 70 how old is my sister?",
	 "context": None},
	{"id": 2, "prompt": "Q: When I was 6 my sister was half my age. Now I'm 70 how old is my sister?\n\nA:",
	 "context": """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done,
there will be 21 trees. How many trees did the grove workers plant today?\n
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted.
So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74
chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops
did Jason give to Denny?\n
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of
lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does
he have now?\n
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so
in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from
monday to thursday. How many computers are now in the server room?\n
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 =
20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers.
The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many
golf balls did he have at the end of wednesday?\n
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On
Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?\n
A: She bought 5 bagels for \$3 each. This means she spent \$15. She has \$8 left.

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

st.subheader("Self-Consistency")
st.markdown("""Perhaps one of the more advanced techniques out there for prompt engineering is self-consistency. \
Proposed by Wang et al. (2022\, self-consistency aims \"to replace the naive greedy decoding used in chain-of-thought prompting\". \
The idea is to sample multiple, diverse reasoning paths through few-shot CoT, and use the generations to select the most consistent answer. \
This helps to boost the performance of CoT prompting on tasks involving arithmetic and commonsense reasoning.""")

text, code = st.columns([0.7, 0.3])


with code:

	with st.container(border=True):
		provider = st.selectbox('provider', helpers.list_providers)
		models = helpers.getmodelIds(provider)
		model = st.selectbox(
			'model', models, index=models.index(helpers.getmodelId(provider)))
		temperature = st.slider('temperature', min_value=0.0, max_value= 1.0, value = st.session_state[suffix]['temperature'], step = 0.1)
		topP = st.slider('top_p', min_value=0.0, max_value=1.0, value = st.session_state[suffix]['topP'], step = 0.1)
		if provider in ["Anthropic","Cohere","Mistral"]:
			topK = st.slider('top_k', min_value=0, max_value=200, value = st.session_state[suffix]['topK'], step = 1)
		maxTokenCount = st.slider('max_tokens', min_value=50, max_value=4096, value = st.session_state[suffix]['maxTokenCount'], step = 100)


with text:

	tab1, tab2 = st.tabs(["No Self-Consistency", "Self-Consistency"])

	with tab1:
		output = helpers.prompt_box(questions[0]['id'], provider,
							model,
       						prompt=questions[0]['prompt'],
							maxTokenCount=maxTokenCount,
							temperature=temperature,
							topP=topP,
       						topK=topK,
							context=None)
		
		if output:
			st.write("### Answer")
			st.info(output)
	with tab2:
		st.markdown(questions[1]['context'])

		output = helpers.prompt_box(questions[1]['id'], provider,
							model,
							prompt=questions[1]['prompt'],
							maxTokenCount=maxTokenCount,
							temperature=temperature,
							topP=topP,
       						topK=topK,
							context=questions[1]['context'])
		
		if output:
			st.write("### Answer")
			st.info(output)

