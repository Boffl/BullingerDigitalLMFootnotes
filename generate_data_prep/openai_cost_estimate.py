import tiktoken
import argparse, os
import jsonlines, json
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
    # example call: python openai_cost_estimate.py instruct_add test gpt-3.5 1.5 2
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_type", choices=["instruct_add", "instruct_contintue"])
    parser.add_argument("split", choices=["train", "dev", "test"])
    parser.add_argument("model_name", help="for the tokenization, no need to be too specific")
    parser.add_argument("price_in_per_M", type=float)
    parser.add_argument("price_out_per_M", type=float)
    parser.add_argument("--example_out_message", default="")
    parser.add_argument("--human_eval", default=False, action="store_true", help="only calculate for the human feedback")
    parser.add_argument("--DATA_DIR", default="../../data")

    args = parser.parse_args()
    prompt_type = args.prompt_type
    split = args.split
    model_name = args.model_name
    price_in_per_M = args.price_in_per_M
    price_out_per_M = args.price_out_per_M
    example_out_message = args.example_out_message
    human_eval = args.human_eval
    DATA_DIR = args.DATA_DIR

    folder_path = os.path.join(DATA_DIR, "prompts", prompt_type, split)

    # Choose your model's encoding, e.g., for GPT-4 models
    encoding = tiktoken.encoding_for_model(model_name)

    

    if human_eval:
        with open(os.path.join(DATA_DIR,"human_feedback_prompts.json"), "r", encoding="utf-8") as injson:
            human_eval_list = json.load(injson)

        for el in human_eval_list:
            if el not in os.listdir(folder_path):
                print(el)
        
        filenames = [filename for filename in os.listdir(folder_path) if filename in human_eval_list]
    else:
        filenames = os.listdir(folder_path)

    tokens = []  # list of tuples, in- and out token count

    for filename in tqdm(filenames):
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

    

