# functions to run over the prompts

import os, csv
from tqdm import tqdm
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
import argparse
import time
import threading
import GPUtil
import logging

global model
global tokenizer

adapter_map = {
  "EA": "pretrain-EA",
  "qa": "instruct-qa",
  "add": "instruct-add",
  "bible": "pretrain-bible",
  "Zwa": "pretrain-Zwa",
  "Z": "pretrain-Z"

}

ONE_SHOT_10224 = [{"role": "user", "content": "Bitte schlage mir einen Text für Fussnote n=3 in folgendem Dokument vor:\n\n<TEI xmlns=\"http://www.tei-c.org/ns/1.0\" xml:id=\"file10224\" type=\"Brief\" source=\"HBBW-3\" n=\"193\">\n\t<teiHeader xml:lang=\"de\">\n\t\t<fileDesc>\n\t\t\t<titleStmt>\n\t\t\t\t<title subtype=\"file\">Konrad Geßner, Johannes Fries / Basel an Heinrich Bullinger, 25. Februar [1533]</title>\n\t\t\t</titleStmt>\n\t\t\t<publicationStmt>\n\t\t\t\t<authority>Universität Zürich</authority>\n\t\t\t\t</publicationStmt>\n\t\t\t<sourceDesc>\n\t\t\t\t</sourceDesc>\n\t\t</fileDesc>\n\t\t</teiHeader>\n\t<text xml:lang=\"la\">\n\t\t<body>\n\t\t\t<div xml:id=\"div1\" corresp=\"regest1\">\n\t\t\t\t<p>\n\t\t\t\t\t<s n=\"1\" xml:lang=\"la\" type=\"auto\">Optimo et integerrimo viro M. Henrico Bullingero, mecaenati charissimo.</s>\n\t\t\t\t</p>\n\t\t\t\t<p>\n\t\t\t\t\t<s n=\"2\" xml:lang=\"la\" type=\"auto\">S.</s>\n\t\t\t\t\t<s n=\"3\" xml:lang=\"la\" type=\"auto\">Impediunt nos ab itinere<note xml:id=\"fn3\" type=\"footnote\" n=\"3\"></note> nives, pluvia et ventorum vis.</s>\n\t\t\t\t\t<s n=\"4\" xml:lang=\"la\" type=\"auto\"><placeName ref=\"l28\" cert=\"high\">Basileae</placeName> apud <persName ref=\"p8418\" cert=\"high\">Myconium</persName><note xml:id=\"fn4\" type=\"footnote\" n=\"4\"></note> sine sumtu moramur sudum coelum et tempestatem mitiorem expectantes.</s>\n\t\t\t\t\t<s n=\"5\" xml:lang=\"la\" type=\"auto\">Nivibus obrutae sunt viae omnes, montes praesertim, per quos nulla itinera nunc patent.</s>\n\t\t\t\t\t<s n=\"6\" xml:lang=\"la\" type=\"auto\">Maxime tamen omnium nos detinet, quod Gallos et alios quosdam itineris comites facturos brevi hic invenimus.</s>\n\t\t\t\t\t<s n=\"7\" xml:lang=\"la\" type=\"auto\">Tuam humanitatem rogamus literas nobis a senatu poscat Tigurinos nos esse et a Tigurino senatu propter studia ablegatos.</s>\n\t\t\t\t\t<s n=\"8\" xml:lang=\"la\" type=\"auto\">Ita enim docti plerique consuluerunt, quo nobis tutioribus esse liceat.</s>\n\t\t\t\t\t<s n=\"9\" xml:lang=\"la\" type=\"auto\">Ne nos negligas etiam atque etiam oramus poscimusque.</s>\n\t\t\t\t\t<s n=\"10\" xml:lang=\"la\" type=\"auto\">Si dederis operam, facile impetrabis.</s>\n\t\t\t\t\t<s n=\"11\" xml:lang=\"la\" type=\"auto\">Literis nos acceptis<note xml:id=\"fn6\" type=\"footnote\" n=\"6\"></note> quamprimum cum comitibus maturabimus iter.</s>\n\t\t\t\t</p>\n\t\t\t\t<p>\n\t\t\t\t\t<s n=\"12\" xml:lang=\"la\" type=\"auto\">Vale et nos tibi commendatos habe.</s>\n\t\t\t\t</p>\n\t\t\t\t<p>\n\t\t\t\t\t<s n=\"13\" xml:lang=\"la\" type=\"auto\"><placeName ref=\"l28\" cert=\"high\">Basileae</placeName> in aedibus <persName ref=\"p8418\" cert=\"high\">Myconii</persName>, februarii 25.</s>\n\t\t\t\t</p>\n\t\t\t\t<p>\n\t\t\t\t\t<s n=\"14\" xml:lang=\"la\" type=\"auto\"><persName ref=\"p1214\" cert=\"high\">Ioannes Frisius</persName> et <persName ref=\"p1283\" cert=\"high\">C. Gesnerus</persName> tui toti.</s>\n\t\t\t\t</p>\n\t\t\t</div>\n\t\t</body>\n\t</text>\n</TEI>\n"},
                    {"role": "assistant", "content": "<persName ref=\"p1283\" cert=\"high\">Geßner</persName> und <persName ref=\"p1214\" cert=\"high\">Fries</persName> befanden sich auf dem Weg nach <placeName ref=\"l59\" cert=\"high\">Bourges</placeName>."}
]

# Function to monitor GPU usage
def monitor_gpu(interval=5):
    while True:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            log_message = (f"GPU {gpu.id} - Load: {gpu.load * 100:.1f}% | "
                           f"Free Memory: {round((gpu.memoryFree/(gpu.memoryFree+gpu.memoryUsed))*100)}% | "
                           f"Temperature: {gpu.temperature}°C")
            logging.info(log_message)
        time.sleep(interval)


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
    
    log_message = f"prompt length: {input_ids.shape[1]}"
    logging.info(log_message)

    # generate the attention mask, got a warning that it might be needed, though I am not sure if that is necessary
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)

    if "llama" in model.config._name_or_path:
      terminators = [
          tokenizer.eos_token_id,
          tokenizer.convert_tokens_to_ids("<|eot_id|>")
      ]

      # enable FlashAttention
      with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
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
      torch.cuda.empty_cache()
      filepath = os.path.join(folder_path, filename)
      with jsonlines.open(filepath) as infile:
        messages = [line for line in infile]
        old_prompt_input_tokens =  tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").shape[1]
        try:
          generated_footnote = generate_chat(messages, model, tokenizer)
        except RuntimeError as e:
          if 'CUDA out of memory' in str(e):
            logging.info("CUDA out of memory")
            # Try to replace the one-shot with the shorter letter
            print(f"letter {letter_id} causes out of memory error, trying with shorter prompt")
            messages[1] = ONE_SHOT_10224[0]  # example user question
            messages[2] = ONE_SHOT_10224[1]  # example answer
            new_prompt_input_tokens = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").shape[1]
            torch.cuda.empty_cache()
            try:
              generated_footnote = generate_chat(messages, model, tokenizer)
            # If it still does not work...
            except RuntimeError as e:
              if 'CUDA out of memory' in str(e):
                logging.info("CUDA out of memory")
                print(f"letter {letter_id} causes out of memory error, even with shorter prompt")
                long_letters_set.add(letter_id)
                torch.cuda.empty_cache()

                # if we have the instruct_add prompt, we can add the window version
                if prompt_type == "instruct_add":
                  substitute_filepath = filepath.replace("instruct_add", "instruct_add_window")
                  with jsonlines.open(substitute_filepath) as infile:
                    messages = [line for line in infile]
                    try: 
                      generated_footnote = generate_chat(messages, model, tokenizer)
                    except RuntimeError as e:
                      if 'CUDA out of memory' in str(e):
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

def load_model(size, adapters=[], quantized=True):
  """load the model with the adapters, if specified into global scope"""
  global model
  global tokenizer
  
  if quantized:
    model_id = f"unsloth/Meta-Llama-3.1-{size}B-Instruct-bnb-4bit"
  else:
    model_id = f"meta-llama/Llama-3.1-{size}B-Instruct"

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

def get_model(size, adapters=[], quantized=True):
  """Returns the model to use in other modules"""
  load_model(size, adapters, quantized)
  return model, tokenizer

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
    monitor_thread = threading.Thread(target=monitor_gpu, args=(5,), daemon=True)
    monitor_thread.start()

  global DATA_DIR
  DATA_DIR = args.dir
  load_model(args.size, args.adapters)
  run_llama_over_prompts(args.prompt, args.split, model_name=f"llama-{args.size}B")

  if args.log_gpu_usage:
    # Stop the monitoring once the main task is done
    stop_monitoring.set()
    monitor_thread.join()

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("size", choices=["8", "70"], help="Size of the llama model, either 8 or 70")
  parser.add_argument("prompt", choices=["instruct_qa", "instruct_add"])
  parser.add_argument("split", choices=["dev_100", "dev", "test", "test_human_eval"])
  parser.add_argument(
    "--adapters",
    nargs="*",
    default=[],
    help="A list of adapters, if multiple are specified, they are combined (default is an empty list)"
  )
  parser.add_argument("--dir", default = "/data/nbauer/data", help="Directory where the data are stored, default=/data/nbauer/data")
  parser.add_argument("--log_gpu_usage", default="", help="Log-file for gpu-usage, no logging if left empty")
  args = parser.parse_args()

  main(args)
