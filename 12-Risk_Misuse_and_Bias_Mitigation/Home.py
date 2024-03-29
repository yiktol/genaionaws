import streamlit as st #all streamlit commands will be available through the "st" alias
from utils import set_page_config


set_page_config()

st.title("Risks and Misuse")

st.markdown("""Prompt hacking is a term used to describe attacks that exploit vulnerabilities of LLMs, by manipulating their inputs or prompts. Unlike traditional hacking, which typically exploits software vulnerabilities, prompt hacking relies on carefully crafting prompts to deceive the LLM into performing unintended actions.

We will cover three types of prompt hacking: prompt injection, prompt leaking, and jailbreaking. Each relates to slightly different vulnerabilities and attack vectors, but all are based on the same principle of manipulating the LLM's prompt to generate some unintended output.

To protect against prompt hacking, defensive measures must be taken. These include implementing prompt based defenses, regularly monitoring the LLM's behavior and outputs for unusual activity, and using fine tuning or other techniques. Overall, prompt hacking is a growing concern for the security of LLMs, and it is essential to remain vigilant and take proactive steps to protect against these types of attacks.
            """, unsafe_allow_html=True)
