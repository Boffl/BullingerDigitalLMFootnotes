import argparse
import os
import pandas as pd
import json
import jsonlines
from tqdm import tqdm

# open the meta data file
def get_letter_ids(split):
    with open("../../data/strat_sample.json", "r", encoding="utf-8") as injson:
        samples = json.load(injson)
    return samples[split]

footnote_df = pd.read_csv("../../data/footnote_downsized_df.csv")


# get the letter, get the sentence with the footnote

def make_prompt(n_footnote, xml_sentence):

    system_prompt = "You are a helpful assistant"
    user_query = f"""Welche Frage beantwortet Fussnote {n_footnote} in diesem Satz? Bitte gebe mir nur eine Frage , die sich die Editorin gestellt hat, bevor sie recherchiert und dese Fussnote geschrieben hat.
    {xml_sentence}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    return messages


# create the prompt for the question

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=str, choices=["train", "dev", "test"])
    parser.add_argument("--example", action="store_true", default=False)  # only do 5 example letters for testing purposes
    args = parser.parse_args()
    split = args.split
    example = args.example

    out_path = f"../../data/prompts/get_questions"
    if example:
        out_path = os.path.join(out_path, "example")
    else:
        out_path = os.path.join(out_path, split)

    letter_ids = get_letter_ids(split)
    if example:
        letter_ids = letter_ids[:5]

    # go over letters
    for letter_id in tqdm(letter_ids):

        letter_df = footnote_df[footnote_df["letter_id"]==int(letter_id)]
        for _, row in letter_df.iterrows():
            messages = make_prompt(row.n_footnote, row.xml_sentence)
            outfile_name = f"{letter_id}_{row.n_footnote}.jsonl"
            outfile_path = os.path.join(out_path, outfile_name)
            with jsonlines.open(outfile_path, "w") as outfile:
                outfile.write_all(messages)
        
