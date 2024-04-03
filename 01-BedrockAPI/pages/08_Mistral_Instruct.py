import utils.stlib as stlib
import utils.mistral as mistral
import streamlit as st

stlib.set_page_config()

suffix = 'mistral'
if suffix not in st.session_state:
    st.session_state[suffix] = {}


dataset = mistral.load_jsonl('data/mistral.jsonl')

stlib.initsessionkeys(dataset[0], suffix)
stlib.initsessionkeys(mistral.params, suffix)


text, code = st.columns([0.7, 0.3])

with code:

    with st.container(border=True):
        provider = st.selectbox('provider', ['Meta'])
        model = mistral.modelId()
        streaming = st.checkbox('Streaming')
    with st.container(border=True):
        params = mistral.tune_parameters()


with text:
    st.title('Mistral AI')
    st.markdown("""Mistral AI is a small creative team with high scientific standards. We make efficient, helpful and trustworthy AI models through ground-breaking innovations.
- A 7B dense Transformer, fast-deployed and easily customisable. Small, yet powerful for a variety of use cases. Supports English and code, and a 32k context window.
- A 7B sparse Mixture-of-Experts model with stronger capabilities than Mistral 7B. Uses 12B active parameters out of 45B total. Supports multiple languages, code and 32k context window.
				""")

    with st.expander("See Code"):
        st.code(mistral.render_mistral_code(
            'instruct.jinja', suffix), language="python")

    tab_names = [f"Prompt {question['id']}" for question in dataset]

    tabs = st.tabs(tab_names)

    for tab, content in zip(tabs, dataset):
        with tab:
            outputs = mistral.prompt_box(content['id'], model,
                                         context=content['prompt'],
                                         height=content['height'],
                                         streaming=streaming,
                                         **params)

            if outputs and not streaming:
                st.write("### Answer")
                for index, output in enumerate(outputs):
                    st.info(f"Output {index + 1}\n")
                    st.success(f"Text:\n{output['text']}\n")
                    st.info(f"Stop reason: {output['stop_reason']}\n")
