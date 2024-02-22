from lxml import etree
from tqdm import tqdm
import os
import sys
import shutil
import csv

infolder = sys.argv[1]
if len(sys.argv)>2:
    outfolder = sys.argv[2]

def get_files_with_footnotes():
    """Copies all letters that contain footnotes into a different folder, to browse and look around"""
    filenames = [f'{i}.xml' for i in range(10013, 13160)]  # edited letters
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

def get_node_text(node):
    """get the string of the xml in a node"""
    text = node.text
    if text:  # sometimes text will be none
        text += ''.join([etree.tostring(sub).decode('utf-8') for sub in node])
    else:
        text = ''.join([etree.tostring(sub).decode('utf-8') for sub in node])

    return text

def put_footnotes_in_csv():
    """get all footnotes into a csv file, to look at them"""

    # define namespace
    namespaces_none = {None: 'http://www.tei-c.org/ns/1.0'}  # works well with findall, but not with .xpath
    namespaces_tei = {'tei': 'http://www.tei-c.org/ns/1.0'}  # when using .xpath
    # open outfile
    with open('all_notes.csv', 'w', encoding='utf-8', newline='') as outfile:

        csv_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True)
        csv_writer.writerow(['letter_id', 'n_footnote', "n_sentence", 'text_footnote', 'text_sentence'])  # column names

        # iterate through all files
        filenames = os.listdir(infolder)
        for filename in tqdm(filenames):
            letter_id = filename.split('.')[0]
            if int(letter_id)<10013:
                continue
            filepath = os.path.join(infolder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = etree.parse(f)
            root = tree.getroot()

            # find footnotes
            # todo: does not find all?
            footnotes = root.findall(".//note[@type='footnote']", namespaces_none)
            if footnotes:
                # add all the footnotes to the element (not the editorial ones)
                for footnote in footnotes:
                    # Note apparently there is this bug that if a closin tag is followed by a white-space char, the
                    # following text is considered a 'tail' of the node and is included. We don't want it so we remove it
                    footnote.tail = None
                    # Check if the attribute n is a number, if not it is an editorial comment and we can move on
                    try:
                        n_footnote = int(footnote.get('n'))
                        text_footnote = get_node_text(footnote)
                    except ValueError:
                        continue

                    # get the sentence
                    try:
                        sentence = root.xpath(f".//tei:s[descendant::tei:note[@n='{n_footnote}']]", namespaces=namespaces_tei)[0]
                    except IndexError:  # no ancestor is a sentence, in this case let's ignore for now
                        continue
                    n_sentence = sentence.xpath("./@n", namespaces=namespaces_tei)[0]
                    text_sentence = get_node_text(sentence)


                    # write the data as a row
                    csv_writer.writerow([letter_id, n_footnote, n_sentence, text_footnote, text_sentence])

def remove_footnotes(infolder, outfolder):
    """takes files from one folder, removes footnotes and saves them in a new folder"""
    filenames = os.listdir(infolder)
    for filename in tqdm(filenames):
        filepath = os.path.join(infolder, filename)
        with open(filepath, 'r', encoding='utf-8') as infile:
            tree = etree.parse(infile)
        root = tree.getroot()
        # find footnotes
        footnotes = root.findall(".//note[@type='footnote']", root.nsmap)
        for footnote in footnotes:
            try:
                float(footnote.get('n'))
            except ValueError:
                continue
            parent = footnote.getparent()
            parent.remove(footnote)

        outpath = os.path.join(outfolder, filename)
        with open(outpath, "w", encoding="utf-8") as outfile:
            outfile.write(etree.tostring(root, pretty_print=True, with_tail=False).decode('utf-8'))



if __name__ == '__main__':
    put_footnotes_in_csv()