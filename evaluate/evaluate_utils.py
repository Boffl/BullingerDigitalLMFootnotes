import re
import pandas as pd
from evaluate import load
from html import unescape
import evaluate
import bert_score
from tqdm import tqdm
import multiprocessing

bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")
###########
# Calculating BERT scores
###########
# Default of what layer to use for the representation
# The original module defines the default as
# "the number of layers tuned on WMT16 correlation data."
# I have to add some models, that are not in the original
# Maybe a todo: Test if the evaluation changes significantly based on this...
bert_score.utils.model2layers["benjamin/roberta-base-wechsel-german"] = 9

class genEvaluator():
  def __init__(self, df, processes=8, batch_size=64):
    self.df = df
    self.processes = processes
    self.batch_size = batch_size
    self.refs = [normalize_ws(ref) for ref in list(df["xml_footnote"])]
    self.preds = [normalize_ws(pred) for pred in list(df["generated_footnote"])]
    self.preds_without = [remove_markup(pred) for pred in self.preds]
    self.refs_without = [remove_markup(ref) for ref in self.refs]
    self.updated_data = dict()
    try: 
      self.bert_with = list(df["bert_with"])
    except KeyError:
      print("bert_with is missing")
    
    try: 
      self.bert_without = list(df["bert_without"])
    except KeyError:
      print("bert_without is missing")
      
    try:
      self.bleu_with = list(df["bleu_with"])
      self.rouge_with = list(df["rouge_with"])
    except KeyError:
      print("No BLEU and ROUGE with markup are missing")
      
    try:
      self.bleu_without = list(df["bleu_without"])
      self.rouge_without = list(df["rouge_without"])
    except KeyError:
      print("No BLEU and ROUGE without are missing")

  def compute_bertscore_with_markup(self):
    print("Computing BERT with Mark-Up")
    self.bert_with = compute_bertscore(self.preds, self.refs, batch_size=self.batch_size)["f1"]
    self.updated_data["bert_with"] = self.bert_with

  def compute_bertscore_without_markup(self):
    print("Computing BERT without Markup")
    self.bert_without = compute_bertscore(self.preds_without, self.refs_without, batch_size=self.batch_size)["f1"]
    self.updated_data["bert_without"] = self.bert_without

  def compute_bleu_rouge_with_markup(self):
    print("calculating BLEU and ROUGE with markup")
    pool = multiprocessing.Pool(processes=self.processes)
    bleu_rouge_results = list(tqdm(pool.imap(compute_bleu_rouge, zip(self.preds, self.refs)), total=len(self.preds)))
    pool.close()
    pool.join()
    self.bleu_with = []
    self.rouge_with = []
    for bleu, rouge in bleu_rouge_results:
       self.bleu_with.append(bleu)
       self.rouge_with.append(rouge)
    self.updated_data["bleu_with"] = self.bleu_with
    self.updated_data["rouge_with"] = self.rouge_with
    
  def compute_bleu_rouge_without_markup(self):
    print("calculating BLEU and ROUGE without markup")
    pool = multiprocessing.Pool(processes=self.processes)
    bleu_rouge_results = list(tqdm(pool.imap(compute_bleu_rouge, zip(self.preds_without, self.refs_without)), total=len(self.preds)))
    pool.close()
    pool.join()
    self.bleu_without = []
    self.rouge_without = []
    for bleu, rouge in bleu_rouge_results:
       self.bleu_without.append(bleu)
       self.rouge_without.append(rouge)
    self.updated_data["bleu_without"] = self.bleu_without
    self.updated_data["rouge_without"] = self.rouge_without


  def compute_bertscore(self):
    self.compute_bertscore_with_markup()
    self.compute_bertscore_without_markup()

  def compute_bleu_rouge(self):
    self.compute_bleu_rouge_with_markup()
    self.compute_bleu_rouge_without_markup()
  
  def update_and_return_df(self):
    for column, values in self.updated_data.items():
      self.df[column] = values
    return self.df
    
      
     
     
     
      


def remove_outer_note_tag(xml_str):
  """remove the outer note tag, since in the footnote_xml in the df it is still contained"""
  match_obj = re.match(r"<note [^>]*?>(.*?)</note>", xml_str, re.DOTALL)
  if match_obj: return match_obj.group(1)
  else: return xml_str

def normalize_ws(text):
  """normalize whitespace"""
  return re.sub(r"\s+", " ", text).strip()

def remove_markup(sent):
  """Remove all markup from a sentence"""
  return unescape(re.sub(rf"<.*?>", "", sent))

def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)




def evaluate_batch(preds, refs, batch_size=64, processes=8):
  """evaluate batch-wise
  Well the BERT-score is used with the batch argument
  Rouge and Bleu, idk how to properly paralellize, for now using multiprocessing"""

  refs = [normalize_ws(ref) for ref in refs]
  preds = [normalize_ws(pred) for pred in preds]

  results =  {
            "bleu":[],
            "rouge":[],
            "bertscore":[]
          }
  print("calculating BLEU and ROUGE...")
  pool = multiprocessing.Pool(processes=processes)
  bleu_rouge_results = list(tqdm(pool.imap(compute_bleu_rouge, zip(preds, refs)), total=len(preds)))
  pool.close()
  pool.join()
  for bleu, rouge in bleu_rouge_results:
    results["bleu"].append(bleu)
    results["rouge"].append(rouge)

  print(f"calculating semantic scores with BERT, batch_size = {batch_size}...")
  results["bertscore"].append(compute_bertscore(preds, refs, batch_size=batch_size)["f1"])
  return results

def evaluate_with_and_without(preds, refs, batch_size=64, processes=8):
  """run evaluation function with and without markup"""
  all_results = {
        "with_markup": {},
        "without_markup": {}
        }
  print("\n###########evaluating with mark-up########\n")
  all_results["with_markup"] = evaluate_batch(preds, refs, batch_size=batch_size, processes=processes)
  print("\n###########evaluating without mark-up########\n")
  preds = [remove_markup(pred) for pred in preds]
  refs = [remove_markup(ref) for ref in refs]
  all_results["without_markup"] = evaluate_batch(preds, refs, batch_size=batch_size, processes=processes)
  

def compute_bleu_rouge(pred_ref_pair):
  pred, ref = pred_ref_pair
  result_bleu = bleu.compute(predictions=[pred], references=[ref])["bleu"]
  result_rouge = rouge.compute(predictions=[pred], references=[ref])["rouge1"]
  return (result_bleu, result_rouge)


def compute_bertscore(predictions, references, batch_size=2):

  results = {"precision": [], "recall": [], "f1": []}
  with tqdm(total=len(predictions)) as pbar:
      for i in range(0, len(predictions), batch_size):
          batch_preds = predictions[i : i + batch_size]
          batch_refs = references[i : i + batch_size]
          batch_results = bertscore.compute(
              predictions=batch_preds, references=batch_refs, lang="de"
          )
          results["precision"].extend(batch_results["precision"])
          results["recall"].extend(batch_results["recall"])
          results["f1"].extend(batch_results["f1"])
          pbar.update(len(batch_preds))
  return results


def evaluate_set(machine_df, human_df, batch_size=64, processes=8):
  merged_df = machine_df.merge(human_df, on=["letter_id", "n_footnote"])
  predictions = list(merged_df["generated_footnote"])
  references = list(merged_df["xml_footnote"])

  all_results = evaluate_with_and_without(predictions, references, batch_size, processes)
  all_results = flatten_dict(all_results)
  result_df = pd.DataFrame(all_results)
  
  return pd.concat([merged_df, result_df], axis=1)

                                      