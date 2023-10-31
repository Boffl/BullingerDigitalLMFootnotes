from lxml import etree
from tqdm import tqdm
import os
import sys
import shutil
import csv

infolder = sys.argv[1]

def get_files_with_footnotes():
    """Copies all letters that contain footnotes into a different folder, to browse and look around"""
    filenames = [f'{i}.xml' for i in range(1, 10014)]  # first 10'013 letters are edited
    # filenames = [f'{i}.xml' for i in range(1, 11)]  # for debugging
    # filenames = ['247.xml']
    footnote_query = ".//note[@type='footnote']"

    for filename in tqdm(filenames):
        filepath = os.path.join(infolder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = etree.parse(f)
        root = tree.getroot()
        footnotes = root.findall(footnote_query, namespaces=root.nsmap)
        if footnotes:
            new_filepath = os.path.join(infolder, '..', 'letters_with_footnotes', filename)
            shutil.copyfile(filepath, new_filepath)

def get_all_footnotes():
    """get all footnotes into a separate xml file"""

    # create root element
    namespaces = {None: 'http://www.tei-c.org/ns/1.0'}
    footnote_root = etree.Element('letters', nsmap=namespaces)

    # iterate through all files
    filenames = os.listdir(infolder)
    for filename in tqdm(filenames):
        letter_id = filename.split('.')[0]
        filepath = os.path.join(infolder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = etree.parse(f)
        root = tree.getroot()

        # find footnotes
        footnotes = root.findall(".//note[@type='footnote']", namespaces)
        if footnotes:
            # create a letter element
            letter_el = etree.Element('letter', nsmap=namespaces)
            letter_el.set('letter_id', letter_id)

            # add all the footnotes to the element (not the editorial ones)
            for footnote in footnotes:
                # Note apparently there is this bug that if a closin tag is followed by a white-space char, the
                # following text is considered a 'tail' of the node and is included. We don't want it so we remove it
                footnote.tail = None

                # Check if the attribute n is a number, if not it is an editorial comment and we can move on
                try:
                    float(footnote.get('n'))
                except ValueError:
                    continue
                letter_el.append(footnote)

            # add the element to the root
            footnote_root.append(letter_el)


    with open('all_notes.xml', 'w', encoding='utf-8') as out:
        out.write(etree.tostring(footnote_root, pretty_print=True, with_tail=False).decode('utf-8'))

    # print(etree.tostring(footnote_root, pretty_print=True).decode('utf-8'))

def put_footnotes_in_csv():
    """get all footnotes into a csv file, to look at them"""

    # define namespace
    namespaces = {None: 'http://www.tei-c.org/ns/1.0'}
    # open outfile
    with open('all_notes.csv', 'w', encoding='utf-8', newline='') as outfile:

        csv_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True)
        csv_writer.writerow(['letter_id', 'n', 'text'])  # column names

        # iterate through all files
        filenames = os.listdir(infolder)
        for filename in tqdm(filenames):
            letter_id = filename.split('.')[0]
            filepath = os.path.join(infolder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = etree.parse(f)
            root = tree.getroot()

            # find footnotes
            footnotes = root.findall(".//note[@type='footnote']", namespaces)
            if footnotes:
                # add all the footnotes to the element (not the editorial ones)
                for footnote in footnotes:
                    # Note apparently there is this bug that if a closin tag is followed by a white-space char, the
                    # following text is considered a 'tail' of the node and is included. We don't want it so we remove it
                    footnote.tail = None
                    # Check if the attribute n is a number, if not it is an editorial comment and we can move on
                    try:
                        n = float(footnote.get('n'))
                        text = footnote.text
                    except ValueError:
                        continue
                    if text:  # sometimes text will be none
                        text += ''.join([etree.tostring(sub).decode('utf-8') for sub in footnote])
                    else:
                        text = ''.join([etree.tostring(sub).decode('utf-8') for sub in footnote])

                    # write the data as a row
                    csv_writer.writerow([letter_id, n, text])

if __name__ == '__main__':
    put_footnotes_in_csv()