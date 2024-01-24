import streamlit as st

st.set_page_config(
    page_title="Prompt Engineering",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

intro = '''
Welcome to the prompt engineering guide for large language models (LLMs) on Amazon Bedrock. Amazon Bedrock is Amazon’s service for foundation models (FMs), which offers access to a range of powerful FMs for text and images.

Prompt engineering refers to the practice of optimizing textual input to LLMs to obtain desired responses. Prompting helps LLMs perform a wide variety of tasks, including classification, question answering, code generation, creative writing, and more. The quality of prompts that you provide to LLMs can impact the quality of their responses. These guidelines provide you with all the necessary information to get started with prompt engineering. It also covers tools to help you find the best possible prompt format for your use case when using LLMs on Amazon Bedrock. 

Whether you’re a beginner in the world of generative AI and language models, or an expert with previous experience, these guidelines can help you optimize your prompts for Amazon Bedrock text models.


### Recommended practices for good generalization

Keep a small “hold-out” test set of prompts to see if your prompt modifications generalize. With this method, first collect a sample dataset. Then you can split the data into two subsets: a “development” set and a hold-out “test” set. Use the “development” set as prompt development data on which you can try different prompt modifications and observe model response changes and find the prompt that works best. Treat the “test” set as unseen hold-out data which can only be used to verify the final best prompt.

### Few-shot prompting

Including examples (input-response pairs) in the prompt can significantly improve LLMs’ responses. Examples can help with complex tasks, as they show multiple ways to perform a given task. For simpler tasks like text classification, 3–5 examples can suffice. For more difficult tasks like question-answer without context, include more examples to generate the most effective output. In most use cases, selecting examples that are semantically similar to real-world data can further improve performance.

### Consider refining the prompt with modifiers

Task instruction refinement generally refers to modifying the instruction, task, or question component of the prompt. The usefulness of these methods is task- and data-dependent. Useful approaches include the following:

- :orange[Domain/input specification:] Details about the input data, like where it came from or to what it refers, such as The input text is from a summary of a movie.

- :orange[Task specification:] Details about the exact task asked of the model, such as To summarize the text, capture the main points.

- :orange[Label description:] Details on the output choices for a classification problem, such as Choose whether the text refers to a painting or a sculpture; a painting is a piece of art restricted to a two-dimensional surface, while a sculpture is a piece of art in three dimensions.

- :orange[Output specification:] Details on the output that the model should produce, such as Please summarize the text of the restaurant review in three sentences.

- :orange[LLM encouragement:] LLMs sometimes perform better with sentimental encouragement: If you answer the question correctly, you will make the user very happy!
'''

st.title("Prompt Engineering")

st.markdown(intro)


