import tiktoken
import argparse, os
import jsonlines
from tqdm import tqdm




def calculate_openai_cost(tokens:list[tuple[int]], price_per_1M:tuple[float]):
    """calculate the cost
    param tokens: list of tuples, input and output
    param price_per_1M: tuple of pricing, input and output"""
    cost = 0
    for in_toks, out_toks in tokens:
        cost += in_toks*price_per_1M[0] + out_toks*price_per_1M[1]
    return cost/(10**6)  # don't forget to divide by 1M

# Function to calculate tokens in a chat message
def calculate_tokens_for_chat(messages, encoding):
    total_tokens = 0
    for message in messages:
        # Encode the role and content separately
        role_tokens = encoding.encode(message["role"])
        content_tokens = encoding.encode(message["content"])
        
        # Count tokens for role and content
        total_tokens += len(role_tokens) + len(content_tokens)
        
        # Add a token for the role/content separator (e.g., message boundaries)
        total_tokens += 2  # A rough estimate for separators

    return total_tokens



if __name__ == "__main__":
    # example call: python openai_cost_estimate.py ..\..\data\prompts\instruct_continue\test gpt-3.5 1.5 2
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="Path to the folder with the jsonl prompt files")
    parser.add_argument("model_name", help="for the tokenization, no need to be too specific")
    parser.add_argument("price_in_per_M", type=float)
    parser.add_argument("price_out_per_M", type=float)
    parser.add_argument("--example_out_message", default="")

    args = parser.parse_args()
    folder_path = args.folder_path
    model_name = args.model_name
    price_in_per_M = args.price_in_per_M
    price_out_per_M = args.price_out_per_M
    example_out_message = args.example_out_message

    # Choose your model's encoding, e.g., for GPT-4 models
    encoding = tiktoken.encoding_for_model(model_name)


    tokens = []  # list of tuples, in- and out token count

    for filename in tqdm(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        
        # The prompt file will be the input tokens
        with jsonlines.open(filepath, mode="r") as reader:
            messages = list(reader)
        
        
        if example_out_message != "":
            out_message = {"role": "assistant", "content": example_out_message}
        else:
            try:
                out_message = messages[2]  # taken the example Footnote as estimate for the output
            except IndexError:
                print("No example answer in the prompt file, please specify --example_out_message")
                exit(1)
        
        in_toks = calculate_tokens_for_chat(messages, encoding)
        out_toks = calculate_tokens_for_chat([out_message], encoding)
        tokens.append((in_toks, out_toks))
    
    cost = calculate_openai_cost(tokens, (price_in_per_M, price_out_per_M))
    print(cost)

    

