import streamlit as st
from utils import helpers


helpers.set_page_config()


text, code = st.columns([0.7,0.3])

with open('data/01_case_study_1.txt') as f:
    prompt = f.read()
   
with open('data/rds_for_oracle.txt') as f:
    context1 = f.read()


questions = ["What is customer trying to accomplish? Write in one paragraph only and nothing more.",
             "What are the most relevant keywords in the case study that highlight the need to migrate to the cloud. Output a list a up to a maximum of ten keywords and nothing more. Do not repeat any reference to this prompt.",
             "What is the main Pain Point or Challenges of the case study? Write at least three challenges in numbered bullet list and don't say anything else. Do not repeat any reference to this prompt.",
             "What are the Qualifying Questions to identify the requirements, constraints and the possible solutions? Display only a maximum of 10 questions in numbered bullet list and nothing more.",
             "Propose a single AWS solution for the challenges of Cathay Pacific Airways and explain it's business value. Select the one fully manage AWS service that will solve most of the challenges of the PROS system.",
             "List the Benefits to Customer on migrating PROS to Amazon RDS for Oracle.",
             "List up to five possible category of objections from Cathay Pacific Airways if we suggest migrating the PROS to Amazon RDS for Oracle. Display your answer in markdown table where the first column is the category and the second column is a one sentence description of the possible solution to the objection."]

if "outputs" not in st.session_state:
    st.session_state.outputs = []

with st.sidebar:
	with st.container(border=True):
		provider = st.selectbox('provider', helpers.list_providers)
		models = helpers.getmodelIds(provider)
		model = st.selectbox(
			'model', models, index=models.index(helpers.getmodelId(provider)))
		
	with st.container(border=True):
		params = helpers.tune_parameters(provider)


st.subheader("Lead Generator")
with st.container(border=True):
    st.markdown(prompt)
submit = st.button("Generate Lead", type='primary')

    
if submit:
    st.session_state.outputs = []
    with st.spinner("Thinking..."):
        
        for question in questions:
            
            context=f"{prompt}\n\n{context1}\n\n{question}"
        
            output = helpers.prompt_box("1", provider,
								model,
								context=context,#\n\n{context1}\n\n{context2}\n\n{context3}",
                                streaming=None,
								**params)
            st.session_state.outputs.append(output)

if st.session_state.outputs:
                 
    with st.container(border=True):
        st.markdown("#### OPPORTUNITY AND QUALIFICATION")
        objective, keywords, challenge =  st.columns([0.4,0.3,0.3])
        objective.info(f"**What is customer trying to accomplish?**:\n\n {st.session_state.outputs[0]}")
        keywords.success(f"**Key Words/Questions**:\n\n {st.session_state.outputs[1]}")
        challenge.info(f"**Pain Point or Challenge**:\n\n {st.session_state.outputs[2]}")
        st.success(f"**Qualifying Questions**:\n\n {st.session_state.outputs[3]}")
        
    with st.container(border=True):
        st.markdown("#### PROPOSED SOLUTION/VALUE")
        description, benefits =  st.columns([0.5,0.5])
        description.info(f"**Description**:\n\n {st.session_state.outputs[4]}")
        benefits.success(f"**Joint Solution Benefits to Customer**:\n\n {st.session_state.outputs[5]}")
        
    with st.container(border=True):
        st.markdown("#### OBJECTION HANDLING")
        st.markdown(f'{st.session_state.outputs[6]}')
