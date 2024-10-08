from lxml import etree
import re, json
from tqdm import tqdm

def main():
    with open("../../data/strat_sample.json", "r", encoding="utf-8") as injson:
        strat_sample = json.load(injson)
    
    letter_ids = strat_sample["train"]
    for letter_id in tqdm(letter_ids):
        with open(f"../../bullinger_source_data/letters/{letter_id}.xml", "r", encoding="utf-8") as infile:
            tree = etree.parse(infile)
        # root = tree.getroot()
    
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Use XPath to find the <title> element and get its text
        title_text = tree.xpath('//tei:title[@subtype="file"]/text()', namespaces=ns)[0]
        summary_text_list = tree.xpath('//tei:summary//text()', namespaces=ns)
        

        # Join the list of text content into a single string
        summary_text = ' '.join([re.sub(r"(\s)\s+", r"\1", s) for s in summary_text_list])

        with open(f"../../data/sources_literature/hbbw_regesten/{letter_id}.txt", "w", encoding="utf-8") as outfile:
            outfile.write(f"{title_text}\n\n{summary_text}")

if __name__ == "__main__":
    main()
