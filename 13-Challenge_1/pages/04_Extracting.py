import streamlit as st
import utils.helpers as helpers


helpers.set_page_config()
task = """#### Fill in the prompt below to re-create the expected output."""
context = """- John Smith
- Emma Johnson
- Michael Davis
- Olivia Williams
- Abbey Thompson"""
output = """
| LastName | FirstName | LastName, FirstName |
| :--- | :--- | :--- |
| Smith | John | Smith, John |
| Johnson | Emma | Johnson, Emma |
| Davis | Michael | Davis, Michael |
| Williams | Olivia | Williams, Olivia |
| Thompson | Abbey | Thompson, Abbey |
"""

task2 = """#### Fill in the prompt below. From the trip report, you want to be able to extract out action items that need to be done."""
context2 = """#### Trip Report:
I am writing to provide a summary of our recent visit to the ABC Synapse facility on [Date]. The purpose of the trip was to gain a deeper understanding of how ABC Synapse builds and sells electric vehicles (EVs) and to explore potential collaboration opportunities.\n
During the visit, we had the opportunity to tour the manufacturing facility, meet with key personnel, and discuss the company's approach to EV production and sales. The following are some key observations and insights from the trip:
- Manufacturing Process: ABC Synapse has a highly automated and efficient production line, utilizing advanced robotics and AI-driven systems to optimize the assembly of their electric vehicles. This has resulted in reduced production times and increased output capacity. Let's make a note here to conduct a detailed analysis of ABC Synapse's manufacturing process to identify best practices and potential areas for improvement in our own production facilities.
- Battery Technology: The company has invested heavily in research and development to create high-performance, long-lasting batteries for their EVs. Their proprietary battery technology offers extended range and faster charging times compared to competitors.
- Sales and Distribution: ABC Synapse has adopted a direct-to-consumer sales model, bypassing traditional dealerships. This allows them to maintain better control over the customer experience and offer competitive pricing.
Given our company's push for EV production, let's schedule a follow-up meeting with ABC Synapse's R&D team to discuss potential collaboration on battery technology and explore opportunities for joint research projects. We will also need to review our current sales and distribution strategies to determine if adopting a direct-to-consumer model similar to ABC Synapse's approach would be beneficial for our company. \n
Please feel free to reach out if you have any questions or require further information.
"""
output2 = """1. Conduct a detailed analysis of ABC Synapse's manufacturing process to identify best practices and potential areas for improvement in our own production facilities.\n
2. Schedule a follow-up meeting with ABC Synapse's R&D team to discuss potential collaboration on battery technology and explore opportunities for joint research projects.\n
3. Review our current sales and distribution strategies to determine if adopting a direct-to-consumer model similar to ABC Synapse's approach would be beneficial for our company.
"""

suffix = 'challenge2'
if suffix not in st.session_state:
	st.session_state[suffix] = {}


questions = [
	{"id":1,"task": task, "context": context, "output": output},
	{"id":2,"task": task2, "context": context2, "output": output2}
]


text, code = st.columns([0.7, 0.3])


with code:
				  
	with st.container(border=True):
		provider = st.selectbox('provider', helpers.list_providers)
		models = helpers.getmodelIds(provider)
		model = st.selectbox('model', models, index=models.index(helpers.getmodelId(provider)))
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
			with st.expander("See Expected Output"):
					st.markdown(content['output'])
			response = helpers.prompt_box(content['id'], provider,
								model,
								context=content['context'],streaming=streaming,
								**params)
			
			if response and not streaming:
				st.write("### Answer")
				st.info(response)
