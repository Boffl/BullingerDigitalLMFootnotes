
import jsonlines
import os, json, argparse

def generate_batch_file(folder_path, model_name):

    dir_path, sample = os.path.split(folder_path)  # sample will be either of test, dev, train or example
    prompt_name = os.path.basename(dir_path)  # reliant on the directory structure
    outfile_name = f"GenBatch_{prompt_name}_{sample}.jsonl"
    outfile_path = os.path.join(dir_path, outfile_name)

    for filename in os.listdir(folder_path):
        id = filename.split(".")[0]  # letter id and footnote
        file_path = os.path.join(folder_path, filename)
        with jsonlines.open(file_path, "r") as reader:
            messages = list(reader)
        request_dict = {
            "custom_id": id, 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": model_name, 
                "messages": messages
                }
            }
        
        with open(outfile_path, "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(request_dict) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path")
    parser.add_argument("model_name")

    args = parser.parse_args()

    generate_batch_file(args.folder_path, args.model_name)
        

