import os
from tqdm import tqdm

DIR = "../../fine_tuning_data/EA"

def main():
    for filename in tqdm(os.listdir(DIR)):
        with open(os.path.join(DIR, filename), "r", encoding="utf-8") as infile:
            name = filename.split(".")[0]
            section = ""
            section_counter = 1
            for i, line in enumerate(infile):
                section += line
                if line == "\n":# start a new section
                    with open(f"../../fine_tuning_data/EA_split/{name}_section_{section_counter}.txt", "w", encoding="utf-8") as outfile:
                        outfile.write(section)
                    section = ""
                    section_counter += 1
                    

if __name__ == "__main__":
    main()
