import argparse
import os


def file_path(string):
    """Check if string is a valid filepath"""
    if os.path.exists(string):
        return string
    else:
        raise NotADirectoryError

# todo: open letter (without footnotes, get the text element)


# todo: send letter to gpt, with a prompt


# todo: save response, extract the footnotes, position etc.



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data",type=file_path, help="Folder with a subfolder 'human' containing letters and a strat_sample.json file")
    parser.add_argument("--dataset", default="dev", choices=["train", "dev", "test"])
    args = parser.parse_args()
    print(args.path_to_data)
    print(args.dataset)
