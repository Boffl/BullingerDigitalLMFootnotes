# functions to run over the prompts

import os, csv
from tqdm import tqdm
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from typing import Literal
import argparse

adapter_map = {
  "EA": "pretrain-EA",
  "qa": "instruct-qa",
  "add": "instruct-add",
  "bible": "pretrain-bible",

}

def count_prompt_tokens(model_id, prompt_type, split):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  data_path = DATA_DIR
  folder_path = os.path.join(data_path, f"prompts/{prompt_type}/{split}")
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  token_len_list = []
  long_letters_set = set()
  for filename in tqdm(os.listdir(folder_path)):
    filepath = os.path.join(folder_path, filename)
    with jsonlines.open(filepath) as infile:
      messages = [line for line in infile]

    # get the input ids
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    )
    if input_ids.shape[1] > 20000:
      letter_id = filename.split("_")[0]
      long_letters_set.add(letter_id)
    token_len_list.append(input_ids.shape[1])
  return token_len_list, long_letters_set

# Generating with a chat model
def generate_chat(messages:list, model, tokenizer):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # generate the attention mask, got a warning that it might be needed, though I am not sure if that is necessary
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)

    if "llama" in model.config._name_or_path:
      terminators = [
          tokenizer.eos_token_id,
          tokenizer.convert_tokens_to_ids("<|eot_id|>")
      ]


      outputs = model.generate(
          input_ids,
          attention_mask=attention_mask,
          max_new_tokens=256,
          eos_token_id=terminators,
          do_sample=True,
          temperature=0.6,
          top_p=0.9,
          pad_token_id=tokenizer.eos_token_id
      )
    else:  # for the Qwen model
        outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )



    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def run_llama_over_prompts(prompt_type, split, long_letters_set=set(), batch_size=0, testrun=False, model_name=""):
  """run over the set specified. Saves in a csv file under model_responses
  Note that if there is already a file with that name it will only add the
  ones that are not generated yet"""

  global DATA_DIR
  global model
  global model_id

  data_path = DATA_DIR
  folder_path = os.path.join(data_path, f"prompts/{prompt_type}/{split}")

  if model_name == "":  # use default name
    model_name = model_id.split("/")[-1]

  if type(model.active_adapter) == str:
    adapter_name = model.active_adapter
  else:
    adapter_name = "base"
  outfile_path = os.path.join(data_path, f"model_responses/llama/{model_name}-{adapter_name}_{prompt_type.replace('_', '-')}_{split.replace('_', '-')}.csv")
  if testrun:
    outfile_path = outfile_path.replace(".csv", "_testrun.csv")
  print(f"writing results to {outfile_path}")

  finished = []  # tuples of letter_id and n_footnote that are already done
  if os.path.exists(outfile_path):
    with open(outfile_path, "r", encoding="utf-8") as infile:
      reader = csv.reader(infile)
      next(reader)  # skip header
      try:
        finished = [(row[0], row[1]) for row in reader]
      except IndexError:
        print("faulty csv file found, rewriting")
        finished = []

  if not finished:  # if the file was non existent or faulty... rewrite
    print("no data yet")
    with open(outfile_path, "w", encoding="utf-8") as outfile:
      outfile.write(f"letter_id,n_footnote,generated_footnote\n")
  else:
      print(f"already finished generations: {len(finished)}")

  unfinished = []
  for filename in os.listdir(folder_path):
    letter_id = filename.split("_")[0]
    n_footnote = filename.split("_")[1].split(".")[0]
    if (letter_id, n_footnote) in finished:
      continue
    # ignore the long letters for if specified
    if letter_id in long_letters_set:
      continue
    unfinished.append((filename, letter_id, n_footnote))
  print(f"total FNs to generate: {len(finished)+len(unfinished)}")

  if batch_size == 0:
    for filename, letter_id, n_footnote in tqdm(unfinished):
      filepath = os.path.join(folder_path, filename)
      with jsonlines.open(filepath) as infile:
        messages = [line for line in infile]
        try:
          generated_footnote = generate_chat(messages, model, tokenizer)
        except RuntimeError as e:
          if 'CUDA out of memory' in str(e):
            print(f"letter {letter_id} causes out of memory error")
            long_letters_set.add(letter_id)
            torch.cuda.empty_cache()
            continue
          else:
              # Raise other exceptions
              raise e
        torch.cuda.empty_cache()
      with open(outfile_path, "a", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        writer.writerow([letter_id, n_footnote, generated_footnote])

  else:
    print("Batching for generation not implemented...")


def make_adapter_id(adapter, size):
  try:
    adapter_id = f"Boffl/BullingerLM-llama3.1-{size}B-{adapter_map[adapter]}"
  except KeyError:
    print(f"Adapter {adapter} is not allowed")
    exit(1)
  return adapter_id

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

def main(args):
  global DATA_DIR
  DATA_DIR = args.dir
  load_model(args.size, args.adapters)
  run_llama_over_prompts(args.prompt, args.split, model_name=f"llama-{args.size}B")

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("size", choices=["8", "70"], help="Size of the llama model, either 8 or 70")
  parser.add_argument("prompt", choices=["instruct_qa", "instruct_add"])
  parser.add_argument("split", choices=["dev_100", "dev", "test"])
  parser.add_argument(
    "--adapters",
    nargs="*",
    default=[],
    help="A list of adapters, if multiple are specified, they are combined (default is an empty list)"
  )
  parser.add_argument("--dir", default = "/data/nbauer/data", help="Directory where the data are stored, default=/data/nbauer/data")
  args = parser.parse_args()

  main(args)