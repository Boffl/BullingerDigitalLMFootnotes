#####
# Create a subset of the stratified sample

import argparse
import os, json, re
import pandas as pd
from tqdm import tqdm

def check_for_bible_ref(text):
    # referencing the bible (and indicated in the xml)
    bible_ref = r"(Vgl\. |Siehe )?<cit[^>]+?type=\"bible\""
    return bool(re.findall(bible_ref, text))

def main(category):
    with open("../../data/strat_sample.json", "r", encoding="utf-8") as infile:
        strat_sample = json.load(infile)
    
 
    footnote_df = pd.read_csv("../../data/footnote_downsized_df.csv")
    new_dict = {
        "train": [],
        "dev": [],
        "test": []
    }
    for split in strat_sample:
        print(f"\nfiltering footnotes for {split}")
        letter_ids = strat_sample[split]

        for letter_id in tqdm(letter_ids):
            letter_df = footnote_df[footnote_df["letter_id"]==int(letter_id)]

            for _, row in letter_df.iterrows():
                if category == "bible":
                    if check_for_bible_ref(row.xml_footnote):
                        new_dict[split].append([int(letter_id), row.n_footnote])
            
    with open(f"../../data/strat_sample_{category}.json", "w", encoding="utf-8") as outjson:
        json.dump(new_dict, outjson)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category", choices=["bible"])
    args = parser.parse_args()

    main(args.category)
                


