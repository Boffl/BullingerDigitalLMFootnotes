from bs4 import BeautifulSoup
import time, re, os, json, requests
from tqdm import tqdm
import sys

DATA_DIR = "../../fine_tuning_data/Z/"

def get_request(url):
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error {response.status_code}")
        print(f"{response.reason}")
        exit()
    return response.text
    return BeautifulSoup(response.text, "html.parser")


def get_work_links(text):
    """Extract links to individual works from the main page."""
    soup = BeautifulSoup(text, "html.parser")
    links = []
    for a_tag in soup.select('a'):
        if "Nr" in a_tag.text:
            link = a_tag.get("href")
            links.append(link)
    return links


def get_work_text(html):
    """returns title(str) and pages(list[str])"""
    soup = BeautifulSoup(html, "html.parser")
    h3 = soup.select("h3")
    h4 = soup.select("h4")
    title = "\n".join(el.text for el in h3+h4)
    pages = soup.find_all(style="text-align: right;")
    pages = [page.findNext("p").text for page in pages]

    return title, pages


def save_file(title, pages, filepath):
    data = {
        "title": title,
        "pages": pages
    }
    with open(filepath, "w", encoding="utf-8") as outjson:
        json.dump(data, outjson)


def main():
    with open(os.path.join(DATA_DIR, "Briefübersicht.htm"), "r", encoding="utf-8") as infile:  # or Werkeübersicht.htm für die Werke
        text = infile.read()
    for i, link in enumerate(tqdm(get_work_links(text))):
        work_page = get_request(link)
        title, pages = get_work_text(work_page)
        save_file(title, pages, os.path.join(DATA_DIR, "letters", f"{i+1}.json"))  # or works instead of letters
    # save_file(title, pages, "junk.json")

if __name__ == "__main__":


    main()