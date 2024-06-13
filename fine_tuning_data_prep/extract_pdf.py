from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os
from tqdm import tqdm


def convert_pdf_to_txt(pdf_path):
    output_string = StringIO()
    with open(pdf_path, 'rb') as fp:
        resource_manager = PDFResourceManager()
        converter = TextConverter(resource_manager, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, converter)
        for page in PDFPage.get_pages(fp, check_extractable=True):
            interpreter.process_page(page)
        converter.close()

    content = output_string.getvalue()
    output_string.close()

    return content

def convert_dir(pdf_dir, txt_dir):
    """create a copy of the subfolders in the pdf_dir (issues) and convert all containing pdfs to txt"""
    
    try: os.mkdir(txt_dir)
    except FileExistsError: pass

    for issue in tqdm(os.listdir(pdf_dir)):
        issue_pdf = os.path.join(pdf_dir, issue)  # issue folder with pdfs
        issue_txt = os.path.join(txt_dir, issue)  # issue folder with txt
        try: os.mkdir(issue_txt)
        except FileExistsError: pass
        for filename_pdf in os.listdir(issue_pdf):
            filename = filename_pdf.split(".")[0]  # filename without the extension
            filepath_pdf = os.path.join(issue_pdf, filename_pdf)
            filepath_txt = os.path.join(issue_txt, f"{filename}.txt")
            if os.path.isfile(filepath_txt): # we have converted this file already
                continue
            # convert
            content = convert_pdf_to_txt(filepath_pdf)
            with open(filepath_txt, "w", encoding="utf-8") as outfile:
                outfile.write(content)
    
        break





if __name__ == "__main__":
    convert_dir("../../fine_tuning_data/Zwingliana/scraped_pdf", "../../fine_tuning_data/Zwingliana/converted_txt")  # converted_txt
    
