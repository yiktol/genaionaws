import json
from helpers import bedrock_runtime_client

bedrock_runtime = bedrock_runtime_client()


# Define our input prompt template for the task.
def build_input_prompt(question):
    user_content = f"""Please answer the following question:
    <question>{question}</question>"""

    messages = [{'role': 'user', 'content': user_content}]
    return messages

# Get completions for each input.
# Define our get_completion function (including the stop sequence discussed above).
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


# Get completions for each question in the eval.
# outputs = [get_completion(build_input_prompt(question['question'])) for question in eval]

# # Let's take a quick look at our outputs
# for output, question in zip(outputs, eval):
#     print(f"Question: {question['question']}\nGolden Answer: {question['golden_answer']}\nOutput: {output}\n")