from run_llama_over_prompts import *
import logging
from itertools import combinations


def test_single_adapters():
        # example file
    filepath = "/data/nbauer/data/prompts/instruct_add_window/dev_100/13134_23.jsonl"
    with jsonlines.open(filepath) as infile:
        messages = [line for line in infile]
    size = 70
    adapters = ["add", "qa", "EA", "bible"]
    model_id = f"unsloth/Meta-Llama-3.1-{size}B-Instruct-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    for adapter in adapters:
        adapter_id = make_adapter_id(adapter, size)
        model = PeftModel.from_pretrained(model, adapter_id, adapter_name=adapter)
        model.set_adapter(adapter)

        generated_text = generate_chat(messages, model, tokenizer)
        logging.info(f"Output with {model.active_adapter} adapter:")
        logging.info(generated_text)

def load_model(size, adapters=[]):
  """load the model with the adapters, if specified into global scope"""
  global model
  global tokenizer
  
  model_id = f"unsloth/Meta-Llama-3.1-{size}B-Instruct-bnb-4bit"

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id)
 
  if adapters:
    # add the adapter(s)
    # add the first adapter
    adapter_id = make_adapter_id(adapters[0], size)
    model = PeftModel.from_pretrained(model, adapter_id, adapter_name=adapters[0])
    
    # if there are ore adapters make weighted adapters
    while len(adapters) > 1: 
      current_adapter = model.active_adapter

      # load another adapter
      additional_adapter = adapters.pop()
      additional_adapter_id = make_adapter_id(additional_adapter, size)
      model.load_adapter(additional_adapter_id, adapter_name=additional_adapter)

      # create a weighted adapter and activate it
      new_adapter = f"{current_adapter}-{additional_adapter}"
      model.add_weighted_adapter([current_adapter, additional_adapter], [1.0, 1.0], adapter_name=new_adapter, combination_type="linear")
      model.set_adapter(new_adapter)

      # delete the old, non-weighted adapter to save space
      model.delete_adapter(current_adapter)
      model.delete_adapter(additional_adapter)

def test_all_adapters():
    filepath = "/data/nbauer/data/prompts/instruct_add_window/dev_100/13134_23.jsonl"
    with jsonlines.open(filepath) as infile:
        messages = [line for line in infile]
    
    size = 70
    adapters = ["add", "qa", "EA", "bible"]
    load_model(size, adapters)
    generated_text = generate_chat(messages,model,tokenizer)
    logging.info(f"Output with {model.active_adapter} adapter:\n{generated_text}")

def test_adapter_combos():
         # example file
    filepath = "/data/nbauer/data/prompts/instruct_add_window/dev_100/13134_23.jsonl"
    with jsonlines.open(filepath) as infile:
        messages = [line for line in infile]
    size = 70
    adapters = ["add", "qa", "EA", "bible"]
    model_id = f"unsloth/Meta-Llama-3.1-{size}B-Instruct-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # load all adapters
    adapter_id = make_adapter_id(adapters[0], size)
    model = PeftModel.from_pretrained(model, adapter_id, adapter_name=adapters[0])
    for adapter in adapters[1:]:
        adapter_id = make_adapter_id(adapter, size)
        model.load_adapter(adapter_id, adapter_name=adapter)
    
    # make adapter combos
    adapter_combos = []
    for r in [2,3]:
        adapter_combos.extend(combinations(adapters, r))
    
    for combo in adapter_combos:
        new_adapter = "-".join(combo)
        model.add_weighted_adapter(list(combo), [1.0]*len(combo), adapter_name=new_adapter, combination_type="linear")
        model.set(new_adapter)
        generated_text = generate_chat(messages, model, tokenizer)
        logging.info(f"Output with {model.active_adapter} adapter:\n{generated_text}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_function", choices=["single_adapters", "all_adapters", "adapter_combo"])
    parser.add_argument("--log_file_name", default="")
    args = parser.parse_args()
    if args.log_file_name == "":
        log_file_name =f"{args.test_function}_test.log"
    else:
        log_file_name = args.log_file_name
    # Configure logging
    logging.basicConfig(
        filename=log_file_name,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if args.test_function == "single_adapters":
        test_single_adapters()
    if args.test_function == "all_adapters":
        test_all_adapters()
    if args.test_function == "adapter_combo":
        test_adapter_combos()

