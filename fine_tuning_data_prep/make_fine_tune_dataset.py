import argparse
import json
import sys, os
import pandas as pd
from tqdm import tqdm
# import other modules, with the path
module_path = os.path.abspath(os.path.join('..', 'generate_data_prep'))
sys.path.append(module_path)
from generate_instruction import SYSTEM_PROMPT, HISTORIAN_PROMPT, instruct_prompt_add, get_letter_text
from generate_prompts_from_questions import get_query 


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
    
def return_row(df, letter_id, n):
    return df[(df["letter_id"]==int(letter_id))&(df["n_footnote"]==int(n))]

def get_question_df(run_full):
    gpt_responses_path = os.path.join(DATA_DIR, "model_responses", "gpt")

    # questions for the train set:
    df = pd.read_csv(os.path.join(gpt_responses_path, "gpt-4o-mini-2024-07-18-get_questions-train.csv"))

    if run_full:
        df2 = pd.read_csv(os.path.join(gpt_responses_path, "gpt-4o-mini-2024-07-18-get_questions-dev.csv"))
        df = pd.concat([df, df2])
    return df



def main(args):
    # go through all the files in the train set
    # add them to the json
    run_full = args.run_full  # bolean
    prompt = args.prompt
    example = args.example

    # get letter ids from the train set (or train and dev set)
    letter_ids = get_letter_ids(run_full)
    if example:
        letter_ids = letter_ids[:2]

    if prompt == "instruct_qa":
        question_df = get_question_df(run_full)

    footnote_df = pd.read_csv(FOOTNOTE_DF_PATH)

    data = []

    for letter_id in tqdm(letter_ids):
        letter_text = get_letter_text(letter_id)

        ns = list(footnote_df[footnote_df["letter_id"] == int(letter_id)].n_footnote)


        for n in ns:
            letter_no_fns, footnote_content = instruct_prompt_add(letter_text, ns, n)
            if prompt == "instruct_add":
                query = HISTORIAN_PROMPT(letter_no_fns, n)
            elif prompt == "instruct_qa":
                sentence = return_row(footnote_df, letter_id, n)["xml_sentence"].values[0]
                question = return_row(question_df, letter_id, n)["generated_footnote"].values[0]
                query = get_query(sentence, n, question)

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
    parser.add_argument("prompt", choices=["instruct_add", "instruct_qa"])
    parser.add_argument("--run_full", action="store_true", default=False, help="run over train an test set?")
    parser.add_argument("--example", default=False, action="store_true", help="take only 1 letters for testing purposes")

    args = parser.parse_args()

    main(args)



