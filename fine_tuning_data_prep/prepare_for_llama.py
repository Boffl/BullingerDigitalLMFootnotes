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

    # texts = prepare_zwingliana("../../fine_tuning_data/Zwingliana/converted_txt")

    # with open("../../fine_tuning_data/zwingliana_llama_test.json", "w", encoding="utf-8") as outjson:
        # json.dump(texts, outjson, indent=4)

    # todo: make version to open only one file
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", type=str, choices=["EA", "Zwingliana"])
    parser.add_argument("outfile_name", type=str, help="name for the jsonfile that is produced")
    parser.add_argument("--folder_path", default="../../fine_tuning_data", help="Path to the folder with the finetuning data")
    parser.add_argument("--one_file", type=str, help="Just processing one file for testing purposes? Put Filepath here", default="")

    args = parser.parse_args()
    domain = args.domain
    outfile_name = args.outfile_name
    folder_path = args.folder_path
    one_file_path = args.one_file

    if domain == "Zwingliana":
        zwingliana_path = os.path.join(folder_path, "Zwingliana", "converted_txt")
        outfile_path = os.path.join(folder_path, outfile_name)

        texts = []
        if one_file_path:
            with open(one_file_path, "r", encoding="utf-8") as infile:
                texts.append({"text": infile.read()})
                # for text in parse_zwingliana_txt_file(infile):
                  #  texts.append({"text": text})
        
        with open(outfile_path, "w", encoding="utf-8") as outjson:
            json.dump(texts, outjson)
        


    


