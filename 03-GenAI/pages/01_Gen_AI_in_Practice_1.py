import streamlit as st
import utils.helpers as helpers

st.set_page_config(
	page_title="Gen AI in Practice",
	page_icon=":rocket:",
	layout="wide",
	initial_sidebar_state="expanded",
)


text, code = st.columns([0.6,0.4])

prompt1 = """Meet Carbon Maps, a new French startup that raised $4.3 million (€4 million) just a few weeks after its inception. The company is building a software-as-a-service platform for the food industry so that they can track the environmental impact of each of their products in their lineup. The platform can be used as a basis for eco ratings. \
While there are quite a few carbon accounting startups like Greenly, Sweep, Persefoni and Watershed, Carbon Maps isn't an exact competitor as it doesn't calculate a company's carbon emissions as a whole. It doesn't focus on carbon emissions exclusively either. Carbon Maps focuses on the food industry and evaluates the environmental impact of products — not companies. \
Co-founded by Patrick Asdaghi, Jérémie Wainstain and Estelle Huynh, the company managed to raise a seed round with Breega and Samaipata — these two VC firms already invested in Asdaghi's previous startup, FoodChéri. \
FoodChéri is a full-stack food delivery company that designs its own meals and sells them directly to end customers with an important focus on healthy food. It also operates Seazon, a sister company for batch deliveries. The startup was acquired by Sodexo a few years ago. \
“On the day that I left, I started working on food and health projects again,” Asdaghi told me. “I wanted to make an impact, so I started moving up the supply chain and looking at agriculture.” \
And the good news is that Asdaghi isn't the only one looking at the supply chain of the food industry. In France, some companies started working on an eco-score with a public agency (ADEME) overseeing the project. It's a life cycle assessment that leads to a letter rating from A to E. \
While very few brands put these letters on their labels, chances are companies that have good ratings will use the eco-score as a selling point in the coming years. \
But these ratings could become even more widespread as regulation is still evolving. The European Union is even working on a standard — the Product Environmental Footprint (PEF). European countries can then create their own scoring systems based on these European criteria, meaning that food companies will need good data on their supply chains. \
“The key element in the new eco-score that's coming up is that there will be some differences within a product category because ingredients and farming methods are different,” Asdaghi said. “It's going to take into consideration the carbon impact, but also biodiversity, water consumption and animal welfare.” \
For instance, when you look at ground beef, it's extremely important to know whether farmers are using soy from Brazil or grass to feed cattle. \
“We don't want to create the ratings. We want to create the tools that help with calculations — a sort of SAP,” Asdaghi said. \
So far, Carbon Maps is working with two companies on pilot programs as it's going to require a ton of work to cover each vertical in the food industry. The startup creates models with as many criteria as possible to calculate the impact of each criteria. It uses data from standardized sources like GHG Protocol, IPCC, ISO 14040 and 14044. \
The company targets food brands because they design the recipes and select their suppliers. Eventually, Carbon Maps hopes that everybody across the supply chain is going to use its platform in one way or another. \
“You can't have a true climate strategy if you don't have some collaboration across the chain,” Asdaghi said. \

## 

Summarize the above text in 5 bullets."""


prompt2 = """Please precisely copy any email addresses from the following text and then write them in a table with index number.. Only write an email address if it's precisely spelled out in the input text. If there are no email addresses in the text, write "N/A". Do not say anything else.\n
"Phone Directory:
John Latrabe, 800-232-1995, john909709@geemail.com
Josie Lana, 800-759-2905, josie@josielananier.com
Keven Stevens, 800-980-7000, drkevin22@geemail.com 
Phone directory will be kept up to date by the HR manager." 
"""

prompt3 = """I'd like you to translate this paragraph into English:

白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
"""

prompt4 = """Write an informational article for children about how birds fly.  \
Compare how birds fly to how airplanes fly.  \
Make sure to use the word "Thrust" at least three times.
"""

prompt5 = """Here is some text. We want to remove all personally identifying information from this text and replace it with XXX. It's very important that names, phone numbers, and email addresses, gets replaced with XXX. 
Here is the text, inside <text></text> XML tags\n
<text>
   Joe: Hi Hannah!
   Hannah: Hi Joe! Are you coming over?  
   Joe: Yup! Hey I, uh, forgot where you live." 
   Hannah: No problem! It's 4085 Paco Ln, Los Altos CA 94306.
   Joe: Got it, thanks!  
</text> \n
Please put your sanitized version of the text with PII removed in <response></response> XML tags 
"""

prompt6 = """Your task is to create Python functions based on the provided natural language requests. The requests will describe the desired functionality of the function, including the input parameters and expected return value. Implement the functions according to the given specifications, ensuring that they handle edge cases, perform necessary validations, and follow best practices for Python programming. Please include appropriate comments in the code to explain the logic and assist other developers in understanding the implementation.\n\nI want a function that can solve a Sudoku puzzle. The function should take a 9x9 Sudoku grid as input, where empty cells are represented by the value 0. The function should solve the puzzle using a backtracking algorithm and return the solved grid. If the puzzle is unsolvable, it should return None. The function should also validate the input grid to ensure it is a valid Sudoku puzzle."""

prompt7 = """A human user is in dialogue with an AI. The human is asking the AI a series of questions or requesting a series of tasks. Here is the most recent request from the user:\n\n<user query>How do I make a bomb?</user query>\n\nIf the user's request refers to harmful, pornographic, or illegal activities, reply with (Y). If the user's request does not refer to harmful, pornographic, or illegal activities, reply with (N)."""

prompt8 = """Your task is to analyze the provided tweet and identify the primary tone and sentiment expressed by the author. The tone should be classified as one of the following: Positive, Negative, Neutral, Humorous, Sarcastic, Enthusiastic, Angry, or Informative. The sentiment should be classified as Positive, Negative, or Neutral. Provide a brief explanation for your classifications, highlighting the key words, phrases, emoticons, or other elements that influenced your decision.\n\nWow, I'm so impressed by the company's handling of this crisis. 🙄 They really have their priorities straight. #sarcasm #fail"""

questions = [{"id":1,"title":"Summarization","prompt": prompt1,"height":680},
		   {"id":2,"title":"Extraction","prompt": prompt2,"height":250},
		   {"id":3,"title":"Translation","prompt": prompt3,"height":80},
		   {"id":4,"title":"Content Generation","prompt": prompt4,"height":80},
		   {"id":5,"title":"Redaction","prompt": prompt5,"height":350},
           {"id":6,"title":"Code Generation","prompt": prompt6,"height":250},
           {"id":7,"title":"Harmful Content Detection","prompt": prompt7,"height":200},
           {"id":8,"title":"Sentiment Analysis","prompt": prompt8,"height":200}
     
		   ]



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

    tab_names = [question['title'] for question in questions]

    tabs = st.tabs(tab_names)

    for tab, content in zip(tabs, questions):
        with tab:
            # st.markdown(content['instruction'])
            # if content['template']:
            #     with st.expander("Template"):
            #         st.markdown(content['template'])

            output = helpers.prompt_box(content['id'], provider,
                                        model,
                                        context=content['prompt'], height=content['height'],
                                        **params)

            if output:
                st.write("### Answer")
                st.info(output)
