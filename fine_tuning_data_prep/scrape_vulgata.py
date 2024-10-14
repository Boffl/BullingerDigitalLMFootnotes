from bs4 import BeautifulSoup
import requests, os
from tqdm import tqdm

html_filename = 'Latin Bible, Biblia Sacra, Vulgatae Editionis, Hetzenauer 1914.htm'
vulgata_folder = "../../fine_tuning_data/vulgata"
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


