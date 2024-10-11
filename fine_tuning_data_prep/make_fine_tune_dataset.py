import argparse
import json
import sys, os
import pandas as pd
from tqdm import tqdm
# import other modules, with the path
module_path = os.path.abspath(os.path.join('..', 'generate_data_prep'))
sys.path.append(module_path)
from generate_instruction import SYSTEM_PROMPT, HISTORIAN_PROMPT, instruct_prompt_add, get_letter_text


# Some variables
LETTER_DIR = "../../data/downsized_letters"
SAMPLE_FILEPATH = "../../data/strat_sample.json"
DATA_DIR = "../../data"
FOOTNOTE_DF_PATH = "../../data/footnote_downsized_df.csv"
OUT_DIR = "../../data/fine_tune_data"

def get_letter_ids(run_full):
    with open(SAMPLE_FILEPATH, "r", encoding="utf-8") as injson:
        strat_sample = json.load(injson)
    if run_full:
        return strat_sample["train"] + strat_sample["dev"]
    else:
        return strat_sample["train"]
    
def get_footnote_df(footnote_df_path):
    return pd.read_csv(footnote_df_path)


def main(args):
    # go through all the files in the train set
    # add them to the json
    run_full = args.run_full  # bolean
    prompt = args.prompt
    example = args.example

    letter_ids = get_letter_ids(run_full)
    if example:
        letter_ids = letter_ids[:2]

    footnote_df = pd.read_csv(FOOTNOTE_DF_PATH)

    data = []

    for letter_id in tqdm(letter_ids):
        letter_text = get_letter_text(letter_id)

        ns = list(footnote_df[footnote_df["letter_id"] == int(letter_id)].n_footnote)

        if prompt == "instruct_add":
            for n in ns:
                letter_no_fns, footnote_content = instruct_prompt_add(letter_text, ns, n)
                query = HISTORIAN_PROMPT(letter_no_fns, n)
                data.append({
                    "instruction": query,
                    "system": SYSTEM_PROMPT,
                    "input": "",  # could be used to put in the letter, but it is now with the instructions...
                    "output": footnote_content
                })
    outjson_path = os.path.join(OUT_DIR, f'{prompt}{"_exampe" if example else ""}_{"train_dev" if run_full else "train"}.json')
    with open(outjson_path, "w", encoding="utf-8") as outjson:
        json.dump(data, outjson)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", choices=["instruct_add"])
    parser.add_argument("--run_full", action="store_true", default=False, help="run over train an test set?")
    parser.add_argument("--example", default=False, action="store_true", help="take only 1 letters for testing purposes")

    args = parser.parse_args()

    main(args)



