import argparse
import os, json
import pandas as pd
import re
from tqdm import tqdm

DATA_PATH = "../../data"
SYSTEM_PROMPT = "You are a research assistant for a historian, specialized on the European reformation working on an edition of the correspondence of Heinrich Bullinger. Given a letter in TEI format, your task is to add text to a footnote."

MESSAGES_TEMPLATE = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "historian", "content": ""}
]

def get_letter_ids(split:str):
    
    with open(os.path.join(DATA_PATH, "strat_sample.json"), "r", encoding="utf-8") as injson:
        strat_sample = json.load(injson)

    return strat_sample[split]

def get_letter_text(letter_id):

    filename = f"{letter_id}.xml"
    filepath = os.path.join(DATA_PATH, "human", filename)

    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read()
    
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "test"])
    parser.add_argument("--example", action="store_true", default=False)  # only do 5 example letters for testing purposes

    args = parser.parse_args()
    split = args.split
    example = args.example

    out_path = os.path.join(DATA_PATH, "prompts", split)
    if example:
        out_path = os.path.join(out_path, "example")
    
    footnote_df = pd.read_csv(os.path.join(DATA_PATH, "footnote_df.csv"))

    # get the letter ids from the split
    letter_ids = get_letter_ids(split)
    if example:
        letter_ids = letter_ids[:5]

    for letter_id in tqdm(letter_ids):

        text = get_letter_text(letter_id)

        fns = footnote_df[footnote_df["letter_id"] == int(letter_id)]

        for i, fn in fns.iterrows():

            n = fn.n_footnote
            # match the text until the footnote begins. Note: include re.DOTALL, bc otherwise it does not match over the newlines
            text_match = re.match(rf'.*type=\"footnote\" n=\"{n}\">', text, re.DOTALL)
            try:
                prompt_text = text_match.group()
            except AttributeError:
                print(f"Problem with FN {n} in letter {letter_id}")

            outfile_name = f"{letter_id}_{n}.txt"
            outfile_path = os.path.join(out_path, outfile_name)
            with open(outfile_path, "w", encoding="utf-8") as outfile:
                outfile.write(prompt_text)


