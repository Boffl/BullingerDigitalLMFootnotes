import pandas as pd
import os, re, csv, json, re
from html import unescape  # from xml.sax.saxutils import unescape ->> does not work properly... :/
from tqdm import tqdm
from lxml import etree
import argparse

def make_id_to_edition_map(infolder):
    id_to_edition = {}
    for _, _, files in os.walk(infolder):  # we don't care about paths and dirs
        for filename in files:
            filename = filename.split(".")[0]  # remove file extension
            edition, _, letter_id = filename.split("_")  # middle name would be the id of the letter in that edition
            edition = re.search(r"(\d\d)(\w\d?)?", edition).group(1)  # 01A1 -> 01 etc.
            id_to_edition[letter_id] = edition
    return id_to_edition


def count_fns_in_letter(root):
    footnotes = root.findall(".//note[@type='footnote']", {None: 'http://www.tei-c.org/ns/1.0'})
    
    ed_footnote_count = 0  # editorial footnotes
    cont_footnote_count = 0  # content footnotes
    for footnote in footnotes:
        try: 
            int(footnote.get('n'))
            cont_footnote_count += 1
        except ValueError:
            ed_footnote_count += 1
    
    return ed_footnote_count, cont_footnote_count

def make_letter_df(infolder, id_to_edition):
    """make some stats about letters, return pandas df"""
    letter_df = pd.DataFrame(columns=["letter_id", "edition", "sent_count", "cont_footnote_count", "ed_footnote_count"])
    namespaces_none = {None: 'http://www.tei-c.org/ns/1.0'}  # works well with findall, but not with .xpath
    total_footnote_count = 0
    # namespaces_tei = {'tei': 'http://www.tei-c.org/ns/1.0'}  # when using .xpath, but need to use prefix in path
    for filename in tqdm(os.listdir(infolder)):

        with open(os.path.join(infolder, filename), "r", encoding="utf-8") as f:
            tree = etree.parse(f)
        root = tree.getroot()
        letter_id = filename.split(".")[0]

        if letter_id not in id_to_edition:  # only work on edited letters...
            continue
        
        ed_footnote_count, cont_footnote_count = count_fns_in_letter(root)
        

        sentences = root.findall(".//s", namespaces_none) # list of sentences
        if len(sentences) == 0:  # empty letters, not published in the edition?
            continue

        letter_df.loc[len(letter_df)] = [letter_id, id_to_edition[letter_id], len(sentences), cont_footnote_count, ed_footnote_count]
        total_footnote_count += cont_footnote_count
    print("total content footnotes: ", total_footnote_count)
    return letter_df


def make_footnote_df(infolder, outfilename, id_to_edition, test_letter = False):  # todo: write test option
    """get all footnotes into a csv file"""  # with a DataFrame it takes to much time...
    namespaces_none = {None: 'http://www.tei-c.org/ns/1.0'}  # works well with findall, but not with .xpath
    namespaces_tei = {'tei': 'http://www.tei-c.org/ns/1.0'}  # when using .xpath, but need to use prefix in path
    counter = 0
    with open(outfilename, 'w', encoding='utf-8', newline='') as outfile:
        csv_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True)
        csv_writer.writerow(['letter_id', 'edition', 'n_footnote', "n_sentence", 'xml_footnote', 'xml_sentence', 'text_footnote', 'text_sentence', 'len_footnote', 'pos_footnote', 'label'])  # column names
        # footnote_df = pd.DataFrame(columns=[['letter_id', 'edition', 'n_footnote', "n_sentence", 'xml_footnote', 'xml_sentence', 'text_footnote', 'text_sentence', 'len_footnote']])
        for filename in tqdm(os.listdir(infolder)):

            letter_id = filename.split(".")[0]

            if letter_id not in id_to_edition:  # only work on edited letters...
                continue
            
            if test_letter:
                if letter_id != test_letter:
                    continue

            with open(os.path.join(infolder, filename), "r", encoding="utf-8") as f:
                tree = etree.parse(f)
            root = tree.getroot()


            # get the footnotes
            footnotes = root.findall(".//note[@type='footnote']", namespaces_none)
            for footnote in footnotes:
                try:
                    n_footnote = int(footnote.get("n"))
                    # footnote.tail = None  # can't remove it here, otherwise it is also gone in the sentence... 
                    xml_footnote = etree.tostring(footnote, encoding="unicode", with_tail=False) # have to specify the encoding argument as such, bc otherwise it returns a byte string... get_node_string(footnote)
                except ValueError:  # editorial footnotes
                    continue

                # get the sentence to the footnote
                sentence = root.xpath(f".//tei:s[descendant::tei:note[@n='{n_footnote}']]", namespaces=namespaces_tei)
                if len(sentence) == 1:
                    sentence = sentence[0]
                else:  # No anscestor is a sentence element, some are in the Regest, some to other footnotes...
                    # print(f"{letter_id}, {n_footnote}")
                    continue

                n_sentence = sentence.get('n')
                sentence.tail = None
                xml_sentence = etree.tostring(sentence, encoding="unicode") # get_node_string(sentence)

                # replace footnote elements with all content by '__<n>'
                xml_sentence_no_fn = footnote_placeholder(xml_sentence)
                # remove all other markup (assumption is that everything not in footnotes is purely for markup reasons)
                text_sentence = remove_markup(xml_sentence_no_fn)
                text_footnote = remove_markup(xml_footnote)
                len_footnote = len_text(text_footnote)
                pos_footnote = footnote_pos(n_footnote, text_sentence)

                # classify the label
                label = classify_footnote(text_footnote, xml_footnote)

                edition = id_to_edition[letter_id]
                csv_writer.writerow([letter_id, edition, n_footnote, n_sentence, xml_footnote, xml_sentence, text_footnote, text_sentence, len_footnote, pos_footnote, label])
                counter += 1
                
        print(f"Total footnotes found: {counter}")
                    


def downsize_tei(root, footnotes_to_keep=[]):
    """ removing stuff...
    Downsize the header: 
        - keep titleStmt
        - change publicationStmt
            - keep authority and sourceDesc
            - in sourceDesc only keep bibl type='scan' 
        - remove provileDesc
        - remove revisionDesc
- remove facsimile
- remove editorial footnotes
- remove footnotes that are not in the filtered footnotes df"""

    namespaces_tei = {'tei': 'http://www.tei-c.org/ns/1.0'}  # when using .xpath, but need to use prefix in path

    # tags to remove completely
    tags_to_remove = ['facsimile', 'profileDesc', 'revisionDesc']
    nodes_to_remove = [root.find(rf'.//tei:{tag_to_remove}', namespaces=namespaces_tei) for tag_to_remove in tags_to_remove]

    # from the publicationStmt only keep authority
    publicationStmt = root.find(r".//tei:publicationStmt", namespaces=namespaces_tei)
    all_subnodes_except_authority = publicationStmt.xpath("*[not(self::tei:authority)]", namespaces=namespaces_tei)
    nodes_to_remove += all_subnodes_except_authority

    # from sourceDesc only keep bibl type="scan"
    sourceDesc = root.find(r".//tei:sourceDesc", namespaces_tei)
    all_subnodes_except_bibl = sourceDesc.xpath(r"*[not(self::tei:bibl[@type='scan'])]", namespaces=namespaces_tei)
    nodes_to_remove += all_subnodes_except_bibl

    # remove editorial and exessively long footnotes
    footnotes = root.xpath(r".//tei:note[@type='footnote']", namespaces=namespaces_tei)

    for footnote in footnotes:
        try:
            n = int(footnote.get('n'))
            # long footnotes:
            if n not in footnotes_to_keep:
                nodes_to_remove.append(footnote)

        except ValueError:  # editorial footnotes
            nodes_to_remove.append(footnote)


    for node in nodes_to_remove:
        # remove the nodes (note: can't remove from root, as .remove only works for direct children)
        if node is not None:
            parent = node.getparent()
            
            # careful about the tail...
            # if the tail contains text, we put it to the preceding element's tail, or to the parent's text
            if node.tail and node.tail.strip() != "":
                preceding_sibling = node.getprevious()
                if preceding_sibling is not None:
                    # by using the str() function tail gets converted into an empty string in case it is None
                    preceding_sibling.tail = str(preceding_sibling.tail or "") + node.tail
                else:
                    parent.text = str(parent.text or "") + node.tail

            # now we can savely remove the node :D
            parent.remove(node)

    return root
    
    



#####################
# Functions to work with the footnotes
####################


def footnote_placeholder(sent):
    """replaces <footnote n='xyz'> and all its contents by '__xyz'
    Thus both editorial and content footnotes get a placeholder"""

    footnote_regex = (r" ?<note [^>]*? type=\"footnote\" n=\"(.*?)\">" # matching the opening tag, capturing the FN number
                      r".*?(?=<\/note>)"  # matching everything up until the closing tag (positive lookup!!)
                      r"<\/note>"  # matching the endtag
    )

    return re.sub(footnote_regex, r'__\1', sent)

def remove_markup(sent):
    # For now I assume that any other xml tags are purely markup
    # need to unescape non-ascii chars that are saved as escape sequences
    return unescape(re.sub(rf"<.*?>", "", sent))

def footnote_pos(n, sent):
    """Given the footnote number (content footnotes), find the position in the sentence
    Position is determined by tokens separated by whitespace"""
    # n = row["n_footnote"]
    # sent = row["text_sentence"]
    for i, word in enumerate(sent.split()):
        if f'__{n}' in word:
            return i
    return "NaN"

def len_text(text):
    # for now, just simple, tokenization
    return len(text.split())

def get_node_string(node):
    """get the string of the xml in a node"""
    text = node.text
    if text:  # sometimes text will be none
        text += ''.join([etree.tostring(sub, with_tail=True).decode('utf-8') for sub in node])
    else:
        text = ''.join([etree.tostring(sub, with_tail=True).decode('utf-8') for sub in node])

    return text


def test_downsize():
    # test if the downsizing works well by looking at examples

    test_letter_id = "13152"
    with open(f"bullinger-korpus-tei-main/bullinger-korpus-tei-main/data/letters/{test_letter_id}.xml", "r", encoding="utf-8") as f:
        tree = etree.parse(f)
        root = tree.getroot()


    new_tree = etree.ElementTree(downsize_tei(root, [5, 6, 7, 9]))  # need to get the list of footnotes, could also open the dataframe...
    new_tree.write("downsize_tei.xml", encoding="utf-8", pretty_print=True)



def classify_footnote(text, xml_str):
    # quoting a dictionary: Schweizer Idiotikon, Grimm, Otto (latin)
    # Note that "Fischer" is used for quoting from "schwäbisches Wörterbuch" AND "Conrrad Gessner 1515-1565". But from all I can tell only in one instance it is the latter...
    dictionary = r"<bibl.*?>(SI|Grimm|Otto|Fischer)</bibl>"
    # referencing another edition
    self_ref = r"<bibl.*?>(HBBW( (II?I?)?V?I?I?I?X?X?)?)</bibl>"

    # referencing the bible (and indicated in the xml)
    bible_ref = r"(Vgl\. |Siehe )?<cit[^>]+?type=\"bible\""


    # indicating a missing source (like a previous letter that is mentioned)
    missing = r"([Uu]nbekannt.|[Nn]icht erhalten.|[Nn]icht auffindbar.|[Nn]icht bekannt.)$"
    # changed to include it anywhere in the FN
    missing = r"([Uu]nbekannt.|[Nn]icht erhalten.|[Nn]icht auffindbar.|[Nn]icht bekannt.)"

    # referencing the same edition
    inner_ref = r"([Ss]iehe( dazu)? (oben|unten)|([Oo]ben|[Uu]nten)|[Vv]gl. (oben|unten))"  # S.O. and S.u. etc, seem not to be used

    # Regex for lexical footnotes, that do not quote any dictionary
    lex_regex = (r"(^="  # starts with =
                "|([A-Za-zäöüÄÖÜß]+,? ?){1,3}[\.!]$"  # no more than 3 words, sometimes ending in ! if a verb in imperative
                "|[A-Za-zäöüÄÖÜß]+: [A-Za-zäöüÄÖÜß]+)"
                "|[^A-ZÖÄÜß]+$")  # no caps in all of the footnote
    
    # biliographien (only the bullinger bibliography that is cited enough to care)
    # actually it is not cited enough to care... I'll take it out...
    bibl = r"<bibl.*?>HBBibl</bibl>"


    label_list = []

    if re.findall(dictionary, xml_str):
        label_list.append("lex_dict")
        # return "lex_dict"

    if re.findall(self_ref, xml_str): 
        label_list.append("self_ref") 
        # return "self_ref"
    
    if re.search(bible_ref, xml_str):
        label_list.append("bible")
        # return "bible"
    
    if re.search(missing, text):
        label_list.append("missing")
        # return "missing"
    
    if re.match(inner_ref, text):
        label_list.append("inner_ref")
        # return "inner_ref"
    
    if re.match(lex_regex, text):
        label_list.append("lex")
        # return "lex"
    
    # if re.findall(bibl, xml_str):
        # label_list.append("bibl")
        # return "bibl"
    
    if len(label_list) < 1:
            
        if len(text.split()) < 6:  # on the text without markup!!
            label_list.append("short")
            # return "short"

        else:
            label_list.append("misc")
            # return "misc"
    return ", ".join(label_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["letter_df", "footnote_df", "id_to_edition_map"])
    parser.add_argument("outfilename", type=str, help="csv filename for the DFs and json filename for the map")
    parser.add_argument("--infolder", default="../../bullinger_source_data/letters" ,help="folder containing the letters")
    parser.add_argument("--id_to_edition_map", default="../../data/id_to_edition_map.json", help="json file mapping the ids to the edition (can be created with this script, if corresponding folder is available)")
    parser.add_argument("--test_letter", default="", help="for testing cases, run only on a specific letter")
    args = parser.parse_args()
    mode = args.mode 
    outfilename = args.outfilename 
    infolder = args.infolder
    id_to_edition_map = args.id_to_edition_map
    test_letter = args.test_letter
    # call the model to make the dataframes (takes a long time when calling through the ipynb somehow...)
    with open(id_to_edition_map, "r", encoding="utf-8") as injson:
        id_to_edition = json.load(injson)

    if mode == "letter_df":
        letter_df = make_letter_df(infolder, id_to_edition)
        letter_df["footnotes_per_sentence"] = letter_df.cont_footnote_count / letter_df.sent_count  # content footnotes per sentence for the stats
        letter_df.to_csv(outfilename)

    elif mode == "footnote_df":
        make_footnote_df(infolder, outfilename, id_to_edition, test_letter=test_letter)

    elif mode == "id_to_edition_map":
        id_to_edition = make_id_to_edition_map(infolder) 
        with open(outfilename, "w", encoding="utf-8") as outjson:
            json.dump(id_to_edition, outjson)
        
        