import time
import requests
import json
import tiktoken
from sys import exit

class OpenAiModels:

    def __init__(self, model_name:str, api_key, org_id=None):
        self.api_key = api_key
        self.model_name = model_name

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        if org_id:
            self.headers["OpenAI-Organization"] = org_id


    def generate(self, messages:list[dict], logit_bias=None, max_tokens=None, top_p=0.1):
        """returns completion time in tokens per second and the generated text
        exits the program if the generation fails
        @:param messages: a list of the chat history
                            [{'role': 'system', 'content': 'you are...'},{'role': 'user', 'content':'say hi'}

        @:param temp, freq_pen: temperature and frequency penalty
        @:return: the text from the reply"""

        if not self.model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]:
            print(f"{self.model_name} not implemented")
            exit(1)

        data = {
            'model': self.model_name,
            'messages': messages,
            'logit_bias': logit_bias,
            'max_tokens': max_tokens,
            'top_p': top_p,
        }

        for i in range(10):
            try:
                response = requests.post('https://api.openai.com/v1/chat/completions', headers=self.headers, data=json.dumps(data), timeout=5)
            except requests.exceptions.Timeout:
                print('timeout')
                time.sleep(1.2 ** i)  # exponential increase time between failed requests, last request waits for approx. 5 seconds
                continue
            if response.status_code == 200:
                result = response.json()

                if not "error" in result:
                    completion_tokens = result["usage"]["completion_tokens"]
                    prompt_tokens = result['usage']['prompt_tokens']
                    content = result["choices"][0]["message"]["content"]

                    return content, prompt_tokens, completion_tokens  # result["choices"][0]["text"]
                else:
                    print("\n\nOpenAI error: ", result["error"])

            elif response.status_code < 500:
                print("\n\nHTTP ERROR:", response.status_code)

                if response.status_code == 400:
                    print(f"\nPrompt:\n{messages}\n############################\n")

                try:
                    error_dict = response.json()
                    print(error_dict)
                except:
                    pass
                exit(1)


            else:  # Serverside errors (>500)
                time.sleep(1.2 ** i)  # exponential increase time between failed requests, last request waits for approx. 5 seconds

        print("##########\nGeneration failed")
        print("\n\nHTTP ERROR:", response.status_code)
        try:
            print(f"Reason: {response.reason}")
        except:
            pass
        exit(1)

def make_logit_bias(inputs: list[ str ], model_name='gpt-3.5-turbo'):
    """
    returns:  1) a dictionary setting all the ids of the corresponding BPE tokens to 100
              2) The highest number of BPE tokens among the inputs (to set max_len when generating)
    """
    logit_bias = {}
    encoder = tiktoken.encoding_for_model(model_name)
    max_toks = 0  # the maximum number of tokens needed for writing the inputs
    for input in inputs:
        encoding = encoder.encode(input)
        max_toks = max(len(encoding), max_toks)
        for tokenid in encoding:
            logit_bias[ tokenid ] = 100

    return logit_bias, max_toks