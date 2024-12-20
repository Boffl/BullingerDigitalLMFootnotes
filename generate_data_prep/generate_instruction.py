import argparse
import os, json, jsonlines
import pandas as pd
import re
from tqdm import tqdm
from lxml import etree

DATA_PATH = "../../data"
# SYSTEM_PROMPT = "You are a research assistant for a historian, specialized on the European reformation working on an edition of the correspondence of Heinrich Bullinger. Given a letter in TEI format, your task is to add text to a footnote."
SYSTEM_PROMPT = "Du bist Historiker und hast dich auf die Reformation spezialisiert. Gerade arbeitest du daran Briefe von Heinrich Bullinger zu edieren. Vervollständige die inhaltlichen Fussnoten."
def HISTORIAN_PROMPT(text, n):  # used in the instruction_add prompt
  return f"Bitte schlage mir einen Text für Fussnote n={n} in folgendem Dokument vor:\n\n{text}"


def write_jsonl(filepath, items:list[dict]):
    with open(filepath, "w", encoding="utf-8") as outfile:
        for item in items:
            json_line = json.dumps(item)
            outfile.write(json_line, "\n")


def footnote_regex(n):
  matching_string = (fr"( ?<note [^>]*? type=\"footnote\" n=\"{n}\">)" # matching group 1: the opening tag
                  r"(.*?(?=<\/note>))"  # matching group 2 everything up until the closing tag (positive lookup!! no matching group!!)
                  r"(<\/note>)"  # matching group 3: the endtag
  )
  return matching_string

def remove_footnote_content(text, n):
  """remove all content from the footnote n="n" """
  return re.sub(footnote_regex(n), r"\1\3", text)

def get_footnote_content(text, n):
  """get the content of a FN"""
  return re.search(footnote_regex(n), text).group(2)

def get_letter_ids(split:str):
    
    with open(os.path.join(DATA_PATH, "strat_sample.json"), "r", encoding="utf-8") as injson:
        strat_sample = json.load(injson)

    return strat_sample[split]

def get_letter_text(letter_id):

    filename = f"{letter_id}.xml"
    filepath = os.path.join(DATA_PATH, "downsized_letters", filename)

    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read()
    
    return text

def instruct_continue_prompt(letter_text:str, n:int):
    """Returns letter until the start-tag of the FN and the FN content"""

    # match the text until the footnote begins. Note: include re.DOTALL, bc otherwise it does not match over the newlines
    text_match = re.match(rf'.*type=\"footnote\" n=\"{n}\">', letter_text, re.DOTALL)
    try:
        text_until_fn = text_match.group()
    except AttributeError:
        print(f"Problem with FN {n} in letter {letter_id}")
    
    fn_content = get_footnote_content(letter_text, n)

    return text_until_fn, fn_content


def instruct_prompt_add(letter_text, all_footnote_ns:list[int], n:int):
    """Returns letter without FNs and FN content of n.

    :param all_footnote_ns: list of ALL footnote_ns
    :param n: footnote
    """

    footnote_content = get_footnote_content(letter_text, n)

    # removing all FNs
    letter_no_fns = letter_text
    for fn_to_remove in all_footnote_ns:  # Maybe we'll have to take care of the labels here...
        letter_no_fns = remove_footnote_content(letter_no_fns, fn_to_remove)

    return letter_no_fns, footnote_content

def get_letter_context(letter_text, n_footnote, window_size=5):
    """get a window of the letter around the sentence with the footnote"""
    root = etree.fromstring(letter_text)
    namespaces_tei = {'tei': 'http://www.tei-c.org/ns/1.0'}
    # get the sentence to the footnote
    sentence = root.xpath(f".//tei:s[descendant::tei:note[@n='{n_footnote}']]", namespaces=namespaces_tei)[0]
    n_sentence = int(sentence.get('n'))

    # Remove sentences based on n attribute
    for sentence in root.xpath('.//tei:s', namespaces=namespaces_tei):
        n_current_sentence = int(sentence.get("n"))
        if n_current_sentence < n_sentence - window_size or n_current_sentence > n_sentence + window_size:
            sentence.getparent().remove(sentence)

    # Remove empty paragraphs
    for paragraph in root.xpath('.//tei:p', namespaces=namespaces_tei):
        if len(paragraph.xpath('./tei:s', namespaces=namespaces_tei)) == 0:  # Check if no <s> elements are left
            paragraph.getparent().remove(paragraph)
    
    # Remove empty divs
    for div in root.xpath('.//tei:div', namespaces=namespaces_tei):
        if len(div.xpath('./tei:p', namespaces=namespaces_tei)) == 0:  # Check if no <p> elements are left
            div.getparent().remove(div)

    # Convert back to string to see result
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def instruct_prompt_add_window(letter_text, all_footnote_ns:list[int], n:int, window_size):
    """like instruct_prompt_add but instead of returning the whole letter, it takes away all sentences outside of window size
    """
    footnote_content = get_footnote_content(letter_text, n)
    letter_context = get_letter_context(letter_text, n, window_size)
    letter_context_no_fns = letter_context
    for fn_to_remove in all_footnote_ns:
        letter_context_no_fns = remove_footnote_content(letter_context_no_fns, fn_to_remove)
    
    return letter_context_no_fns, footnote_content




if __name__ == "__main__":

    # Example call: python generate_instruction.py test instruct_continue --example
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "test"])
    parser.add_argument("prompt_type", choices=["continue", "instruct_continue", "instruct_add", "instruct_add_window"])  # todo: add continue prompt...
    parser.add_argument("--window_size", default=10, type=int, help="window size for instruct_add_window, default=10")
    parser.add_argument("--example", action="store_true", default=False)  # only do 5 example letters for testing purposes

    args = parser.parse_args()
    split = args.split
    prompt_type = args.prompt_type
    example = args.example

    out_path = os.path.join(DATA_PATH, "prompts", prompt_type, split)
    if example:
        out_path = os.path.join(DATA_PATH, "prompts", prompt_type, "example")
    
    footnote_df = pd.read_csv(os.path.join(DATA_PATH, "footnote_downsized_df.csv"))

    # get the letter ids from the split
    letter_ids = get_letter_ids(split)
    if example:
        letter_ids = letter_ids[:5]

    for letter_id in tqdm(letter_ids):

        letter_text = get_letter_text(letter_id)

        # Footnote numbers in this letter
        ns = list(footnote_df[footnote_df["letter_id"] == int(letter_id)].n_footnote)
        example_n = ns.pop(0)

        if prompt_type == "continue":
            for n in ns:
                text_until_fn, _ = instruct_continue_prompt(letter_text, n)

                outfile_name = f"{letter_id}_{n}.txt"
                outfile_path = os.path.join(out_path, outfile_name)
                with open(outfile_path, "w", encoding="utf-8") as outfile:
                    outfile.write(text_until_fn)

        else:

            # make the 1-shot example
            if prompt_type == "instruct_continue":
                example_query, example_answer = instruct_continue_prompt(letter_text, example_n)
            
            if prompt_type == "instruct_add":
                all_footnote_ns = [example_n] + ns  # The function will remove the ns passed, so we need to pass a list with all of them

                # will have to do this only once, bc letter_no_fns can be used again
                letter_no_fns, example_answer = instruct_prompt_add(letter_text, all_footnote_ns, example_n)
                example_query = HISTORIAN_PROMPT(letter_no_fns, example_n)

            if prompt_type == "instruct_add_window":
                all_footnote_ns = [example_n] + ns
                letter_context_no_fns, example_answer = instruct_prompt_add_window(letter_text, all_footnote_ns, example_n, 10)
                example_query = HISTORIAN_PROMPT(letter_context_no_fns, example_n)

            

            one_shot = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': example_query},
                {'role': 'assistant', 'content': example_answer}
            ]


            for n in ns:

                if prompt_type == "instruct_continue":
                    query, _ = instruct_continue_prompt(letter_text, n)
                
                if prompt_type == "instruct_add":
                    query = HISTORIAN_PROMPT(letter_no_fns, n)

                if prompt_type == "instruct_add_window":
                    letter_context_no_fns, _ = instruct_prompt_add_window(letter_text, all_footnote_ns, n, 10)
                    query = HISTORIAN_PROMPT(letter_context_no_fns, n)

                messages = one_shot + [{'role': 'user', 'content': query}]


                # what would be a better way to save the files??
                # maybee as a big json?
                # or a long jsonl file, every 4 lines starts a new letter. Convenient, but I'd loose metadata
                # So json or many smaller files is better...
                outfile_name = f"{letter_id}_{n}.jsonl"
                outfile_path = os.path.join(out_path, outfile_name)
                with jsonlines.open(outfile_path, "w") as outfile:
                    outfile.write_all(messages)


