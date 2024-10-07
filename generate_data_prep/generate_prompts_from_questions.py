import pandas as pd
import os
from generate_instruction import remove_footnote_content, get_footnote_content
import jsonlines
import argparse
from tqdm import tqdm

DATA_PATH = "../../data/"

SYSTEM_PROMPT = "Du bist Historiker und hast dich auf die Reformation spezialisiert. Gerade arbeitest du daran Briefe von Heinrich Bullinger zu edieren. Vervollständige die inhaltlichen Fussnoten."
def QUESTION_PROMPT(text, n, q):  # used in the instruction_add prompt
  return f"Schlage einen Text für Fussnote {n} vor, die folgende Frage beantwortet: {q} \n\n {text}"

def get_query(sentence, n_footnote, question):
    sentence_removed_fn = remove_footnote_content(sentence, n_footnote)
    return QUESTION_PROMPT(sentence_removed_fn, n_footnote, question)


def save_file(letter_id, n, messages, out_path):
    outfile_name = f"{letter_id}_{n}.jsonl"
    outfile_path = os.path.join(out_path, outfile_name)
    with jsonlines.open(outfile_path, "w") as outfile:
        outfile.write_all(messages)

def make_one_shot_example(example_row):
    """from a row in the df, return messages for one shot"""
    question = example_row["generated_footnote"]
    sentence = example_row["xml_sentence"]
    n_footnote = example_row["n_footnote"]

    query = get_query(sentence, n_footnote, question)
    example_answer = get_footnote_content(sentence, n_footnote)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": example_answer}
    ]
    return messages


def main(split):   
    footnote_df = pd.read_csv("../../data/footnote_downsized_df.csv")

    # get the generated questions
    gpt_response_folder = "../../data/model_responses/gpt"
    filename = f"gpt-4o-mini-2024-07-18-get_questions-{split}.csv"
    filepath = os.path.join(gpt_response_folder, filename)  
    question_df = pd.read_csv(filepath)

    # define where to save the prompt files
    out_path = f"../../data/prompts/instruct_qa/{split}"

    merged_df = pd.merge(footnote_df, question_df, on=["letter_id", "n_footnote"])
    letter_ids = list(merged_df["letter_id"])

    for letter_id in tqdm(letter_ids):
        letter_df_iter = merged_df[merged_df["letter_id"]==letter_id].iterrows()

        # for the one-shot example
        _, example_row = next(letter_df_iter)
        
        
        # add the rest
        for _, row in letter_df_iter:

            # redo the one-shot example
            messages = make_one_shot_example(example_row)

            question = row["generated_footnote"]
            sentence = row["xml_sentence"]
            n_footnote = row["n_footnote"]
            
            query = get_query(sentence, n_footnote, question)
            messages.append({
                "role": "user", "content": query
            })
            save_file(letter_id, n_footnote, messages, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "test", "example"])
    args = parser.parse_args()
    main(args.split)