import streamlit as st
import utils.helpers as helpers

helpers.set_page_config()

task1 = "#### Below are some feedback responses about a course. Use the AI to determine if the course is well-received or not."
context1 = """- Feedback 1 - "Mr. Baker Ree's cooking course was a great experience! The course was affordable, and the cooking school had excellent equipment, especially the very cool oven. I attended other courses before that charged an arm and a leg for just learning one or two recipes. This one had ten!"
- Feedback 2 - "Hygiene - 1 star. Food variety - 4 stars. Trainer - 0 stars."
"Mr. Baker Ree's cooking course was a great value for the money. The school had modern equipment, and people are generally nice. The location at Orchard also make it very accessible. We also learned how to make a wide variety of dishes."
- Feedback 3 - "Aiyo … I didn't enjoy Mr. Baker Ree's cooking course. The trainer laughs at us when we make mistakes. Can like that meh? Very unprofessional. Ask him questions, he also never answer. Then we don't know so when he ask, I cannot answer. Then I got laughed at. Siao. Won't go again. "
- Feedback 4 - "Mr. Baker Ree's cooking course was fantastic! What impressed me most was the brand-new Gaggenau ovens and appliances. Make me feel like a diva cooking for celebrities."
- Feedback 5 - "I loved Mr. Baker Ree's cooking course! The price was unbeatable, and the school had some of the best cooking equipment I've seen. The course mates were also amazing, and we had a blast cooking together."
- Feedback 6 - "I wouldn't recommend Mr. Baker Ree's cooking course. He didn't seem to take hygiene seriously, and sometimes I saw him talking over the food and even spitting while talking. It really turned me off from the course."
- Feedback 7 - “Great investment! The school was modern and well-equipped, and I made some fantastic friends in the course. We also learned so many recipes that I can't wait to try at home!"
- Feedback 8 - "Bad. Too much talking. Too little cooking."
- Feedback 9 - “testing, testing”
"""
output1 = """Group 1 (Positive): Feedback 1, Feedback 4, Feedback 5, Feedback 7
- Affordable course with a good number of recipes taught
- Excellent and modern equipment, including Gaggenau ovens and appliances
- Great value for money
- Accessible location at Orchard
- Wide variety of dishes taught
- Friendly course mates and enjoyable cooking experience
- Good investment, with many recipes to try at home

Group 2 (Negative): Feedback 3, Feedback 6, Feedback 8
- Poor hygiene, with the trainer talking and spitting over the food
- Unprofessional trainer who laughs at mistakes and doesn't answer questions
- Too much talking and not enough cooking

Group 3 (Neutral): Feedback 2, Feedback 9
- Feedback 2 provides mixed opinions, with positive remarks on food variety and equipment, but negative comments on hygiene and the trainer
- Feedback 9 is a test and does not provide any relevant information
"""

task2 = "#### Classify which feedback goes to which department for follow-up"
context2 = """- Feedback 1 - I was walking past Block 52 last night. I stepped onto a banana peel and fell down. There were a lot of broken glass bottles on the floor which cut me. Please do something about this, as a lot of people are drinking in the area, and leaving behind their stuff.
- Feedback 2 - There is constant drilling sounds from late midnight to 3am. It is coming from my neighbour above. Can you do something about it?
- Feedback 3 - I was coming home last night and something felt funny. I thought my gas stove was on, but it wasn't. I leaned out of my window, and I think there's some weird stench coming from the floor below mine. May be a decomposing corpse. Can you investigate?
- Feedback 4 - There is a strong stench coming from the rubbish chute for the past week. Is nobody clearing it?
- Feedback 5 - A lot of people are leaving their beer bottles all around the void deck. It is very unsightly.
- Feedback 6 - I hear sounds of bouncing balls from the floor above. They are very inconsiderate. What can I do?

"""
output2 = """Group 1 (Littering issues - Department L):
- Feedback 1: Banana peel and broken glass bottles on the floor
- Feedback 5: Beer bottles left around the void deck

Group 2 (Smell issues - Department S):
- Feedback 3: Weird stench coming from the floor below
- Feedback 4: Strong stench coming from the rubbish chute

Group 3 (Noise issues - Department N):
- Feedback 2: Constant drilling sounds from late midnight to 3am
- Feedback 6: Sounds of bouncing balls from the floor above
"""
info2 = """- Littering issues go to Department L
- Smell issues go to Department S
- Noise issues go to Department N"""


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

	tab1, tab2 = st.tabs(['Question 1','Question 2'])

	with tab1:
		st.markdown(task1)
		st.markdown(context1)      
		with st.expander("See Expected Output"):
				st.markdown(output1)
		output = helpers.prompt_box(questions[0]['id'], provider,
							model,
							context=questions[0]['context'],
							**params)
		
		if output:
			st.write("### Answer")
			st.info(output)
	with tab2:
		st.markdown(task2)
		st.markdown(context2)
		with st.expander("See Additional Information"):
				st.markdown(info2)  
		with st.expander("See Expected Output"):
				st.markdown(output2)
		
		output = helpers.prompt_box(questions[1]['id'], provider,
							model,
							context=questions[1]['context'],
							**params)
		
		if output:
			st.write("### Answer")
			st.info(output)
