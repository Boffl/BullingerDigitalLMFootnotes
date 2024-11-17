from bs4 import BeautifulSoup
import requests, os, re
from tqdm import tqdm
import argparse
import jsonlines

vulgata_folder = "../../fine_tuning_data/vulgata"

def scrape():
    """scrape the website sacredbible.org"""
    html_filename = 'Latin Bible, Biblia Sacra, Vulgatae Editionis, Hetzenauer 1914.htm'
    # Load the HTML content of the page
    with open(os.path.join(vulgata_folder, html_filename), 'r', encoding='windows-1252') as file:
        content = file.read()

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')

    # Find all the links to the books
    book_links = soup.find_all('a', href=True)

    # Extract the book titles and corresponding URLs
    book_list = []
    base_url = 'http://www.sacredbible.org/vulgate1914/'

    for link in book_links:
        title = link.text.strip()
        href = link['href']
        rel_link = href.split("/")[-1]
        
        if rel_link.startswith('VT-') or rel_link.startswith('NT-'):
            full_url = base_url + href.split('/')[-1]
            book_list.append((title, full_url))

    # Display extracted books and URLs
    print(f"Found {len(book_list)} books")


    def download_book(url, book_name):
        response = requests.get(url)
        filename = f"{book_name}.html"
        if response.status_code == 200:
            
            # Decode the response using windows-1252 :/
            content = response.content.decode('windows-1252')
            
            # Replace the charset meta tag with UTF-8 :)
            content = content.replace(
                'charset=windows-1252',
                'charset=utf-8'
            )
            # save with utf-8 encoding :D
            with open(os.path.join(vulgata_folder, "books", filename), 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            print(f"Failed to download {book_name}")

    for book in tqdm(book_list):
        download_book(book[1],book[0])


def extract_verses_from_html(filepath):
    """Extract bible verses from a html file"""
    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    book_name = os.path.basename(filepath).split(".")[0]
    print(book_name)

    verse_regex = r'\{(\w|\d+):(\d+)\}\s*(.*)'  # some chapters are not a digit (Prologus in Ecclesiasticus)
    
    verses = []
    current_chapter = None
    
    # Iterate over all the text nodes in the body to extract chapters, verses, and content
    for tag in soup.find_all(['a', 'br']):
        if tag.name == 'a' and 'class' in tag.attrs and 'chapter' in tag['class']:
            # Look for the parent <a> tag and extract the chapter number
            if 'name' in tag.parent.attrs:
                current_chapter = tag.parent['name']  # Extract chapter number from 'name' attribute
        elif tag.name == 'br':
            text = tag.previous_sibling
            if text and '{' in text:
                chapter, verse_num, verse_text = re.search(verse_regex, text).groups()

                if chapter != current_chapter:  # just a sanity check
                    print(f"There was a problem in parsing (Chapter numbers don't align).\nBook: {book_name}\ncurrent chapter: {current_chapter}, matched capter: {chapter} \nText: {text}")
                    exit(1)

                verse_data = {
                    'book': book_name,
                    'chapter': current_chapter,
                    'verse': int(verse_num),
                    'text': verse_text.strip()
                }
                verses.append(verse_data)
    
    return verses


# Function to iterate over the HTML files in a folder and write the JSONL file
def generate_jsonl_from_folder(input_folder, output_file):
    book_counter = 0
    verse_counter = 0
    with jsonlines.open(output_file, mode='w') as writer:
        for filename in os.listdir(input_folder):
            if filename.endswith('.html'):
                book_counter += 1
                file_path = os.path.join(input_folder, filename)
                verses = extract_verses_from_html(file_path)
                verse_counter += len(verses)
                for verse in verses:
                    writer.write(verse)
    return book_counter, verse_counter

def parse():
    """parse the verses in the downloaded html files into one jsonl file"""
    input_folder = os.path.join(vulgata_folder, "books")
    outfile_name = "vulgata_bible.jsonl"
    outfile_path = os.path.join(vulgata_folder, outfile_name)
    num_books, num_verses = generate_jsonl_from_folder(input_folder, outfile_path)
    print(f"Saved {num_books} books for a total of {num_verses} verses, Amen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["scrape", "parse"])
    args = parser.parse_args()

    if args.mode == "scrape":
        scrape()
    
    if args.mode == "parse":
        parse()