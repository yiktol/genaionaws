import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()

task1 = """##### Marketing Department is organising a Tech Conference in 3 months time. Several speakers have agreed to come and speak. 

Generate the following:
- Qn 1 - Overall tagline for the event
- Qn 2 - Synopsis for each of the key speakers
- Qn 3 - Marketing description for the event"""
context1 = """Guest of Honour: Dr Xavier McCarthy, Minister for Technology, Country X
Venue: Big Tech Convention Centre Lvl 3\n
Date: 20 Nov 2023\n
Time: 9am to 5pm\n

_Keynote Speaker 1_\n
Alan T, Director of AI from ABC Tech\n
Topics:\n
- Recent advancements in video analytics technology
- How video analytics can be applied in various industries
- Role of AI in developing new video analytics products and solutions

_Keynote Speaker 2_\n
Bernard Shaylor, Head of Data from DEF Synapse, a deep neural networks company\n
Topics:\n
- Discuss the advancements in deep neural networks
- How they're driving innovation in data analysis
- Potential impact on industries, academia, and government decision-making processes

_Keynote Speaker 3_\n
Caroline B. Simmons, Senior Vice President of Sustainablity from GHI Robotics\n
Topics:\n
- Highlight the growing importance of sustainability in the robotics industry
- Strategies for integrating eco-friendly design and materials into the development and production of robotic systems.

Other speakers will be talking on a broad range of topics in AI, Robotics, Deep Learning, Responsible AI, Sustainability, etc.

Partner companies include:
- ABC Tech is an AI company building video analytics solutions
- DEF Synapse is an AI company building innovative Deep Neural Networks
- GHI Robotics is a robotics company dealing in manufacturing robotics
- JKL Innovations is a new web3 company focusing on blockchain and crypto technology,
- MNO Solutions is a consultancy company,
- PQR Automation is a new EV company rising up to challenge Tesla,
- STU Dynamics is a robotics company building next-gen multi-purpose robots
- VW Technologies is a consultancy company for all things tech, especially Data Science and AI,
- XYZ Secure Tech is a consultancy company focusing on cybersecurity, penetration testing, and virus & malware prevention.
"""
output1 = """
"""

questions = [
	{"id":1,"task": task1, "context": context1, "output": output1},
	{"id":2,"task": "Under Construction...", "context": "", "output": ""},
]

text, code = st.columns([0.7, 0.3])


with code:
				  

	with st.container(border=True):
		provider = st.selectbox('provider', helpers.list_providers)
		models = helpers.getmodelIds(provider)
		model = st.selectbox('model', models, index=models.index(helpers.getmodelId(provider)))
	with st.container(border=True):
		params = helpers.tune_parameters(provider)

with text:

	tab_names = [f"Question {question['id']}" for question in questions]

	tabs = st.tabs(tab_names)

	for tab, content in zip(tabs,questions):
		with tab:
			st.markdown(content['task'])
			with st.expander("Additional Information"):
				st.markdown(content['context'])
			output = helpers.prompt_box(content['id'], provider,
								model,
								context=content['context'],
								**params)
			
			if output:
				st.write("### Answer")
				st.info(output)
