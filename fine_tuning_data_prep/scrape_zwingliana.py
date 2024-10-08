
# TODO: rearrange hardcoded paths, to the new directory...

import requests, hashlib
from bs4 import BeautifulSoup
import time, re, os, json
from tqdm import tqdm

# change according to need
FINE_TUNE_DATA_DIR = "../../fine_tuning_data/Zwingliana"  # path where the fine-tuning data is kept
SAVE_PDF_DIR = "junk"  # name of the directory to save the pdfs to


def get_request(url):
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error {response.status_code}")
        print(f"{response.reason}")
        exit()
    
    return response


def download_pdf(pdf_url, filepath):
    # Send a GET request to the URL
    response = requests.get(pdf_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode
        with open(filepath, 'wb') as file:
            # Write the content of the response (the PDF) to the file
            file.write(response.content)

    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
        print(f"filepath: {filepath}")
        print(f"url: {pdf_url}")


def save_html(html_content, description=""):
    """Function to save a HTML to see why something has not worked in scraping"""
    current_time = time.localtime()
    filename = time.strftime(f"%y_%m_%d-%H_%M-{description}.html", current_time)
    with open(filename, "w", encoding="utf-8") as outfile:
        outfile.write(str(html_content))


# filenames will be a hash of the title
try:
    with open("title_mapping.json", "r", encoding="utf-8") as injson:
        title_mapping = json.load(injson)

except FileNotFoundError:
    title_mapping = {}


if __name__ == "__main__": 
    # making sure we are in the right folder, and the top directory is specified correctly (so we don't create a rendom dir, somewhere)
    os.chdir(FINE_TUNE_DATA_DIR)
    try: os.mkdir(SAVE_PDF_DIR)
    except FileExistsError:
        pass
    os.chdir(SAVE_PDF_DIR)

    for a in ["2", "3"]: # Archive pages ["", "2", "3"]
        url = f"https://www.zwingliana.ch/index.php/zwa/issue/archive/{a}"

        response = get_request(url)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        if soup is None:
            print("Error: BeautifulSoup constructor returned None.")
            # check what was wrong with the file
            save_html(response.content, "badness")

        # iterate over all the issues on the archive page
        issues = soup.find_all("a", class_="title")
        for issue in tqdm(issues):
            issue_title = issue.text.strip()
            if "elektronisch noch nicht zugänglich" in issue_title:
                continue

            # make a dir with the issue title
            try:
                issue_title = issue_title.replace("/", "_")  # since 19/2 (1993) is not a suitable folder name...
                os.mkdir(issue_title)
            except FileExistsError:  # we have already downloaded this issue :D
                continue

            issue_url = issue.get("href")

            # get the articles
            response = get_request(issue_url)
            issue_soup = BeautifulSoup(response.content, 'html.parser')
            # Find the div that contains an <h2> element with the text "Article"
            h2_element = issue_soup.find('h2', text=re.compile(r'\s*Artikel\s*'))

            if h2_element is None:
                print("Article element not found...")
                save_html(response.content, "no_article")
                break

            # Find the parent div of the <h2> element
            div_with_articles = h2_element.find_parent('div')

            # get the list of articles
            ul_el = div_with_articles.find("ul")
            li_els = ul_el.find_all("li", recursive=False)  # recursive=False ensures that only direct descendants are found
            for li_el in li_els:
                links = li_el.find_all("a")  # finds links to the articles and the pdf
                
                article_title = links[0].text.strip()
                try:
                    article_url = links[1].get("href")
                except IndexError:  # article most likely has no provided pdf...
                    # print(f"No article url found for \"{article_title}\" in {issue_title}")
                    continue

                # filename = article_title + ".pdf"
                # some article filenames are very long and not suited for
                title_hash = hashlib.sha1(article_title.encode()).hexdigest()
                title_mapping[title_hash] = article_title

                filename = title_hash + ".pdf"

                filepath = os.path.join(issue_title, filename)

                response = get_request(article_url)
                article_soup = BeautifulSoup(response.content, 'html.parser')
                download_el = article_soup.find("a", class_='download')
                pdf_url = download_el.get("href")

                download_pdf(pdf_url, filepath)

                # save the mapping thus far
                with open("title_mapping.json", "w", encoding="utf-8") as outjson:
                    json.dump(title_mapping, outjson)
                




        
