import json
from helpers import bedrock_runtime_client
import re
import streamlit as st
bedrock_runtime = bedrock_runtime_client()

# We start by defining a "grader prompt" template.
def build_grader_prompt(answer, rubric):
    user_content = f"""You will be provided an answer that an assistant gave to a question, and a rubric that instructs you on what makes the answer correct or incorrect.
    
    Here is the answer that the assistant gave to the question.
    <answer>{answer}</answer>
    
    Here is the rubric on what makes the answer correct or incorrect.
    <rubric>{rubric}</rubric>
    
    An answer is correct if it entirely meets the rubric criteria, and is otherwise incorrect. =
    First, think through whether the answer is correct or incorrect based on the rubric inside <thinking></thinking> tags. Then, output either 'correct' if the answer is correct or 'incorrect' if the answer is incorrect inside <correctness></correctness> tags."""

    messages = [{'role': 'user', 'content': user_content}]
    return messages

# Now we define the full grade_completion function.


def get_completion(messages,model="anthropic.claude-3-sonnet-20240229-v1:0",max_tokens=2048,temperature=0.5,top_k=100,top_p=0.9):
    body = {"max_tokens": max_tokens, 
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop_sequences": ["\\n\\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages}    

    accept = 'application/json'
    contentType = 'application/json'
    
    response = bedrock_runtime.invoke_model(body=json.dumps(body), # Encode to bytes
                                    modelId=model, 
                                    accept=accept, 
                                    contentType=contentType)

    response_body = json.loads(response.get('body').read())


    return response_body.get('content')[0]['text']


def grade_completion(output, golden_answer):
    messages = build_grader_prompt(output, golden_answer)
    completion = get_completion(messages)
    st.info(completion)
    # Extract just the label from the completion (we don't care about the thinking)
    pattern = r'<correctness>(.*?)</correctness>'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Did not find <correctness></correctness> tags.")