import streamlit as st
import utils.helpers as helpers


helpers.set_page_config()

questions = [
	{"id": 1, "task": "#### Pick one of the quotes below and translate it into another language that you're familiar with.",
	 "context": """ - A rose by any other name would smell as sweet. By William Shakespeare
- That's one small step for a man, a giant leap for mankind. By Neil Armstrong
- Speak softly and carry a big stick. By Theodore Roosevelt
- Life is like riding a bicycle. To keep your balance, you must keep moving. By Albert Einstein"""},
	{"id": 2, "task": "#### We want a 6-year-old to be able to understand this.",
	 "context": """Digitalisation is a key pillar of the Government's public service transformation efforts. The
Digital Government Blueprint (DGB) is a statement of the Government's ambition to better
leverage data and harness new technologies, and to drive broader efforts to build a digital
economy and digital society, in support of Smart Nation.\n
Our vision is to create a Government that is “Digital to the Core, and Serves with Heart”. A
Digital Government will be able to build stakeholder-centric services that cater to citizens'
and businesses' needs. Transacting with a Digital Government will be easy, seamless and
secure. Our public officers will be able to continually upskill themselves, adapt to new
challenges and work more effectively across agencies as well as with our citizens and
businesses.\n
Two years after the launch of the DGB, the Government has introduced new policies and
initiatives. COVID-19 has also reaffirmed our emphasis on capability building, and
compelled different parts of the Government to accelerate the use of data and of
technology to offer digital services that minimise physical contact, and to use technology
and digital tools to keep us safe.\n
The DGB has been updated to accurately reflect the current plans and to push for more
ambitious goals to pursue deeper and more extensive digitalisation within the
Government. New examples are included to better explain the latest efforts and benefits
of Digital Government. The refresh is in line with the approach to improve the blueprint
iteratively. """},
	{"id": 3, "task": "#### The text below is to be printed in the papers tomorrow. Find out what typos it may have, and correct them.",
	 "context": """##### Product Transummariser is launching today!\n
We are excited to annouce the lauch of our new product that revoluzionizes the way
meetings are conducted and documented. Our product transcribes meetings in real-time
and provides a concize summary of the discussion points, action items, and decisions
made during the meeting. This summary is then automaticaly emailed to all team
members, ensuring that everyone was on the same page and has a clear understanding
of what was discused. With this product, meetings are more effecient, productive, and
inclusive, allowing teams to focus on what really matters - achieving their goals."""}

]


suffix = 'challenge1'
if suffix not in st.session_state:
	st.session_state[suffix] = {}

text, code = st.columns([0.7, 0.3])

with code:

	with st.container(border=True):
		provider = st.selectbox('provider', helpers.list_providers)
		models = helpers.getmodelIds(provider)
		model = st.selectbox(
			'model', models, index=models.index(helpers.getmodelId(provider)))
		if 'AI21' not in provider:
			streaming = st.checkbox('Streaming',value=True)
		else:
			streaming = False
		
	with st.container(border=True):
		params = helpers.tune_parameters(provider)

with text:

	tab_names = [f"Question {question['id']}" for question in questions]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,questions):
		with tab:
			st.markdown(content['task'])
			st.markdown(content['context'])

			output = helpers.prompt_box(content['id'], provider,
								model,
								context=content['context'],streaming=streaming,
								**params)
			
			if output and not streaming:
				st.write("### Answer")
				st.info(output)


