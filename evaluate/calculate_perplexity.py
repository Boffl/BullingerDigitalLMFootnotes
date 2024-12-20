import sys, os, json
import math, csv, re
import torch
import time
import argparse
import GPUtil
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import logging, threading
from datasets import Dataset
from torch.utils.data import DataLoader
from evaluate_utils import remove_outer_note_tag
from huggingface_hub import HfApi, HfFolder
# getting functions from other modules
module_path = "../generate_FNs_llama"
sys.path.append(module_path)
from run_llama_over_prompts import get_model

module_path = "../data"


global model
global tokenizer

# Function to monitor GPU usage
def monitor_gpu(interval, stop_monitoring):
    while not stop_monitoring.is_set():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            log_message = (f"GPU {gpu.id} - Load: {gpu.load * 100:.1f}% | "
                           f"Free Memory: {round((gpu.memoryFree/(gpu.memoryFree+gpu.memoryUsed))*100)}% | "
                           f"Temperature: {gpu.temperature}°C")
            logging.info(log_message)
        time.sleep(interval)

def contains_bible(xml_footnote):
    """return true if footnote contains bible reference"""
    bible_ref = r"(Vgl\. |Siehe )?<cit[^>]+?type=\"bible\""
    return bool(re.search(bible_ref, xml_footnote))

def contains_source(xml_footnote, source):
    """return true if source is quoted in the reference"""
    quotes = re.findall(rf"<bibl.*?>{source}</bibl>", xml_footnote)
    return bool(quotes)

def get_data(data_path, filter=""):
    """get the fns from the dev set in a list
    filter can be either 'bible', 'EA', 'Zwa' or 'Z'"""
    footnote_df = pd.read_csv(os.path.join(data_path, "footnote_downsized_df.csv"))

    with open(os.path.join(data_path, "strat_sample.json"), "r", encoding="utf-8") as injson:
        strat_sample = json.load(injson)
    dev_set_ids = [int(id) for id in strat_sample["dev"]]
    dev_set_FNs = footnote_df[footnote_df["letter_id"].isin(dev_set_ids)]["xml_footnote"]

    if filter:
        if filter not in ["bible", "EA", "Zwa", "Z"]:
            print("Filter not valid, skipping filtering")
        else: 
            orig_len = len(dev_set_FNs)
            if filter == "bible":
                dev_set_FNs = [fn for fn in dev_set_FNs if contains_bible(fn)]
            else:
                dev_set_FNs = [fn for fn in dev_set_FNs if contains_source(fn, filter)]
            
            filtered_len = len(dev_set_FNs)

            print(f"Number of rows before filtering: {orig_len}")
            print(f"Number of rows after filtering: {filtered_len}")



    return [remove_outer_note_tag(fn) for fn in dev_set_FNs]


def tokenize_data(texts, tokenizer):

    # Apply tokenization
    print("tokenizing dataset")
    return [tokenizer(text) for text in texts]


def calculate_perplexity(model, tokenizer, dataset, total=3405):
    total_log_likelihood = 0.0
    total_length = 0

    with torch.no_grad():
        for batch in tqdm(dataset, total=total):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Get logits from the model
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

            # Shift logits and labels for language modeling (causal shift)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()

            # Calculate log probabilities using cross-entropy for each token
            # without averaging to keep precise control
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # Reshape logits to (N, vocab_size)
                shift_labels.view(-1),                         # Reshape labels to (N,)
                reduction='none'                              # No reduction for per-token loss
            )

            # Reshape to original sequence dimensions and apply attention mask
            loss_per_token = loss_per_token.view(shift_labels.size())  # Shape: (batch_size, sequence_length - 1)
            log_likelihood = -(loss_per_token * shift_attention_mask).sum().item()
            non_padding_tokens = shift_attention_mask.sum().item()

            # Accumulate total log likelihood and token count
            total_log_likelihood += log_likelihood
            total_length += non_padding_tokens

        # Calculate average log likelihood and perplexity
        avg_log_likelihood = total_log_likelihood / total_length
        perplexity = math.exp(-avg_log_likelihood)
    
    return perplexity


def get_dataloader(tokenized_dataset, batch_size, tokenizer):

    def collate_fn(batch):
        return tokenizer.pad(batch, padding="longest", return_tensors="pt")
    
    return DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    

def store_ppl(model_name, adapter_name, batch_size, len_dataset, ppl):
    with open("ppl_eval.csv", "a", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        writer.writerow([f"{model_name}-{adapter_name}", batch_size, len_dataset, ppl])



def main(args):
    if args.log_gpu_usage:
        # Configure logging
        logging.basicConfig(
            filename=args.log_gpu_usage,
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Define a stop event for clean exit
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(target=monitor_gpu, args=(5,stop_monitoring), daemon=True)
        monitor_thread.start()

    model, tokenizer = get_model(args.size, args.adapters)

    model_name=f"llama-{args.size}B"
    if type(model.active_adapter) == str:
        adapter_name = model.active_adapter
    else:
        adapter_name = "base"

    # testing dataset:
    dataset = get_data(args.dir, filter=args.filter_dataset)
    if args.test:
        dataset = dataset[:args.test]
    print("len dataset: ", len(dataset))
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    dataloader = get_dataloader(tokenized_dataset, args.batch_size, tokenizer)
    
    ppl = calculate_perplexity(model, tokenizer, dataloader, total=math.ceil(len(dataset)/args.batch_size))
    print(ppl)
    
    store_ppl(model_name, adapter_name, args.batch_size, len(dataset),  ppl)

    if args.log_gpu_usage:
        print("stop monitoring")
        # Stop the monitoring once the main task is done
        stop_monitoring.set()
        monitor_thread.join()
        sys.exit()
    


if __name__=="__main__":
    ### example call: python calculate_perplexity.py 8 --batch_size 1 --log_gpu_usage test.log
  parser = argparse.ArgumentParser()
  parser.add_argument("size", choices=["8", "70"], help="Size of the llama model, either 8 or 70")
  parser.add_argument(
    "--adapters",
    nargs="*",
    default=[],
    help="A list of adapters, if multiple are specified, they are combined (default is an empty list)"
  )
  parser.add_argument("--batch_size", type=int, default=1, help="heavily affects score with quantized models, though")
  parser.add_argument("--filter_dataset", default="", choices=["bible", "EA", "Zwa", "Z"])
  parser.add_argument("--dir", default = "/data/nbauer/data", help="Directory where the data are stored, default=/data/nbauer/data")
  parser.add_argument("--log_gpu_usage", default="", help="Log-file for gpu-usage, no logging if left empty")
  parser.add_argument("--test", default=0, type=int, help="run only on a partition of the data")
  parser.add_argument("--quantized", type=bool, default=True, help="Set to false to load the original model from Meta")
  args = parser.parse_args()

  main(args)
    
