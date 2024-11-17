import os, json, re
import jsonlines
from tqdm import tqdm
import argparse

OUT_DIR = "../../data/fine_tune_data"



def make_bilbe_dataset(outfile_name, test=False):
    infile_path = "../../fine_tuning_data/vulgata/vulgata_bible.jsonl"
    with jsonlines.open(infile_path) as f:
        verses = list(el for el in f)
    if test:
        verses = verses[:100]
        outfile_name = "test-" + outfile_name

    out_data = []

    # write the first verse 
    verse = verses[0]
    chapter_title = current_chapter_title = " ".join([verse["book"], verse["chapter"]])
    chapter_text = chapter_title + "\n\n" + f"{verse['verse']}: {verse['text']}\n"
    for verse in tqdm(verses[0:]):
        chapter_title = " ".join([verse["book"], verse["chapter"]])
        if current_chapter_title == chapter_title:
            # add the verse
            chapter_text += f"{verse['verse']}: {verse['text']}\n"
        else: # new chapter
            # save the chapter
            out_data.append({
                "text": chapter_text
            })
            # switch to new title
            current_chapter_title = chapter_title
            # start new chapter text
            chapter_text = chapter_title + "\n\n"
            # add the first verse
            chapter_text += f"{verse['verse']}: {verse['text']}\n"
    
    with open(os.path.join(OUT_DIR, outfile_name), "w", encoding="utf-8") as outjson:
        json.dump(out_data, outjson)

def make_zwingilana_dataset(outfilename, test=False):
    infolder = "../../fine_tuning_data/Zwingliana/converted_txt"
    out_data = []
    folders = os.listdir(infolder)
    if test:
        folders = folders[:10]
        outfilename = "test_" + outfilename
    # subfolders per issue
    for folder in tqdm(folders):
        issue_year_regex = r"(\d+(_\d+)?(-\d)?) \((\d{4}(_\d{4})?)\)"
        match_obj = re.search(issue_year_regex, folder)
        if match_obj is None:
            print(f"faulty title: {folder}")
            continue
        issue = match_obj.group(1)
        if "_" in issue:
            issue = issue.replace("_", ",")

        year = match_obj.group(4)
        if "_" in year:
            year = year.replace("_", "-")

        for filename in os.listdir(os.path.join(infolder, folder)):
            with open(os.path.join(infolder, folder, filename), "r", encoding="utf-8") as infile:
                text = infile.read()
            

            out_data.append({
                "text": f"Auschnitt aus der Zwingliana {issue} ({year}): \n\n {text}"
            })


            pages = text.split('\f')
            if len(pages) == 1:
                print(f"file {filename} in {folder} has no pagebreaks?")

            for page in pages:
                out_data.append({
                    "text": f"Auschnitt aus der Zwingliana {issue} ({year}): \n\n {page}"
                })
    
    with open(os.path.join(OUT_DIR, outfilename), "w", encoding="utf-8") as outjson:
        json.dump(out_data, outjson)


def make_EA_dataset(outfile_name, test=False):
    infolder = "../../fine_tuning_data/EA_split"

    out_data = []

    filenames = os.listdir(infolder)
    if test:
        filenames = filenames[:100]

    for filename in tqdm(filenames):
        years_regex = r"\d{4}-\d{4}"
        years = re.search(years_regex, filename).group(0)
        
        with open(os.path.join(infolder, filename), "r", encoding="utf-8") as f:
            text = f.read()
        out_data.append({
            "text": f"Eidgenössische Abschiede in den Jahren {years}\n\n{text}"
        })
    with open(os.path.join(OUT_DIR, outfile_name), "w", encoding="utf-8") as outjson:
        json.dump(out_data, outjson)



def main(domain, test):
    if domain == "bible":
        outfile_name = "pretrain_bible.json"
        make_bilbe_dataset(outfile_name, test)
    elif domain == "EA":
        outfile_name = "pretrain_EA.json"
        make_EA_dataset(outfile_name, test)
    elif domain == "Zwa":
        outfile_name = "pretrain_Zwa.json"
        make_zwingilana_dataset(outfile_name, test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices={"bible", "EA", "Zwa"})
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    main(args.domain, args.test)
            
