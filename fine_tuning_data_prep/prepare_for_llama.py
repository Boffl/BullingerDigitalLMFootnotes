import argparse
import os, json
from tqdm import tqdm




def parse_zwingliana_txt_file(infile):
    """Iterator, returns every page, defined by pagebrake char that comes from converting pdf (U+000c)"""
    text = ""
    for line in infile:
        if line.startswith(b'\x0c'.decode()) and text:
            yield text
            text = line
        else: text += line


def prepare_zwingliana(folderpath):
    """Iterate over the issues and return a dict with the texts
    args: folderpath: Path to the folder with the txt files"""
    
    texts = []
    for path, _, files in tqdm(os.walk(folderpath), total=214):
        for file in files:
            print(path, file)
            try:
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    texts.append({"text": infile.read()})
                    # for text in parse_zwingliana_txt_file(infile):
                    #     texts.append({"text": text})
            except UnicodeDecodeError:
                print("problem with the decoding in file: ", os.path.join(path, file))
    return texts










if __name__ == "__main__":

    texts = prepare_zwingliana("../../fine_tuning_data/Zwingliana/converted_txt")

    with open("../../fine_tuning_data/zwingliana_llama_test.json", "w", encoding="utf-8") as outjson:
        json.dump(texts, outjson, indent=4)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("domain", type=str, choices=["EA", "Zwingliana"])
    
    # parser.add_argument("--path", default="../../fine_tuning_data", help="Path to the folder with the finetuning data")

    # args = parser.parse_args()
    # domain = args.dopain
    # path = args.path

    


