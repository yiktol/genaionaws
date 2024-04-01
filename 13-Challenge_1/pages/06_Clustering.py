import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()

task1 = "#### Get the AI to form meaningful groups from the folowwing list."
context1 = """1) "The Great Gatsby" by F. Scott Fitzgerald
2) "Thriller" by Michael Jackson
3) "To Kill a Mockingbird" by Harper Lee
4) "Giraffe"
5) "1984" by George Orwell
6) “My Heart Will Go On” by Celine Dion
7) "Dolphin"
8) "Elephant"
9) " Hey Jude” by The Beatles
10) " Cosmos by Carl Sagan
11) "The Hobbit" by J.R.R. Tolkien
12) "Cheetah"
13) "Octopus"
14) "Hotel California" by The Eagles
15) “Penguin”
"""
output1 = """Group 1:
- Giraffe
- Dolphin
- Elephant
- Cheetah
- Octopus
- Penguin

Group 2:
- Thriller by Michael Jackson
- My Heart Will Go On by Celine Dion
- Hotel California by The Eagles
- Hey Jude by The Beatles

Group 3:
- The Great Gatsby by F. Scott Fitzgerald
- To Kill a Mockingbird by Harper Lee
- 1984 by George Orwell
- Cosmos by Carl Sagan
- The Hobbit by J.R.R. Tolkien
"""

task2 = "#### Company Meeting Reports: Create a prompt that will re-create the expected output."
context2 = """Your company spoke to several potential partners in tech. The following call reports are recorded after two weeks of meetings.
1. ABC Tech: Our representative met with ABC Tech to discuss the integration of their video analytics solutions into our organization's security framework. Although their AI-driven technology shows potential in enhancing surveillance systems, concerns were raised regarding the privacy implications of implementing such solutions. Both parties agreed to conduct further research to address these concerns before moving forward with any collaboration. We also asked them to show documentation of compliance to local laws.
2. DEF Synapse: Our team met with DEF Synapse to explore the application of their innovative Deep Neural Networks for enhancing our data analysis processes. While their AI technology appeared promising, the complexity of integrating it into our existing systems raised concerns about the feasibility and required resources. They did not seem to support implementation on the cloud. Both parties agreed to continue discussions but will re-evaluate the practicality of this potential partnership.
3. GHI Robotics: We engaged with GHI Robotics to discuss the potential incorporation of their cutting-edge manufacturing robotics into our production facilities. Unfortunately, the high costs associated with the implementation of their advanced robotic systems led to doubts about the overall return on investment. When we probed them about the recent incident where the robot malfunctioned and crushed a worker’s arm, they quickly downplayed that as a one-off unfortunate accident, and the bugs have been fixed. To be cautious, our team decided to postpone any collaboration and explore alternative solutions.
4. JKL Innovations: We had a fruitful meeting with JKL Innovations to discuss the potential implementation of blockchain and crypto technology within our existing infrastructure. They shared that they have gotten the go-ahead from authorities all over the world like US, China, Dhubai, Australia to operate their business. The company shared their expertise in web3 solutions, and we agreed to conduct additional research and analysis to determine the feasibility of this partnership, in particular, integrating them into payment systems here in Singapore.
5. MNO Solutions: Our team met with MNO Solutions to explore potential consulting services aimed at enhancing our business processes and overall strategy. Despite the company's wide-ranging experience in providing tailored solutions, their proposed consulting fees appeared to be significantly higher than our budgetary constraints allowed. We will be reconsidering our options and seeking alternative service providers.
6. PQR Automation: Our representative had a productive meeting with PQR Automation to discuss the possibility of collaborating on electric vehicle (EV) development projects. However, the company's limited track record in the EV industry raised concerns about their ability to effectively challenge established giants like Tesla. While their website and marketing materials boasted of big investors funding them, they were unable to reveal any big names to us citing confidentiality clauses. We agreed to monitor their progress before committing to any partnership.
7. STU Dynamics: We met with STU Dynamics to investigate the opportunities for integrating their next-generation robotics technology into our organization's operations. The company showcased their advanced robotic systems, and we are enthusiastic about the prospects of this collaboration. Further evaluations will be conducted to determine the most effective approach.
"""
output2 = """Group 1 (Follow-up Actions):

1. ABC Tech: Integration of video analytics solutions; concerns about privacy implications; further research needed; documentation of compliance requested.
2. JKL Innovations: Implementation of blockchain and crypto technology; additional research and analysis required; exploring integration into payment systems.
3. STU Dynamics: Integration of next-generation robotics technology; further evaluations needed to determine the most effective approach.

Group 2 (Lower Interest to Proceed):

4. DEF Synapse: Application of Deep Neural Networks; concerns about complexity, feasibility, and required resources; no support for cloud implementation; re-evaluating practicality.
5. GHI Robotics: Incorporation of manufacturing robotics; high costs and doubts about return on investment; concerns about recent robot malfunction incident; exploring alternative solutions.
6. MNO Solutions: Consulting services for business processes and strategy; proposed fees higher than budget; seeking alternative service providers.
7. PQR Automation: Collaboration on electric vehicle development; concerns about limited track record and ability to challenge established giants; unable to reveal big investors; monitoring progress before committing.
"""

questions = [
	{"id":1,"task": task1, "context": context1, "output": output1},
	{"id":2,"task": task2, "context": context2, "output": output2}
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
			st.markdown(content['context'])
			with st.expander("See Expected Output"):
					st.markdown(content['output'])
			response = helpers.prompt_box(content['id'], provider,
								model,
								context=content['context'],
								**params)
			
			if response:
				st.write("### Answer")
				st.info(response)
