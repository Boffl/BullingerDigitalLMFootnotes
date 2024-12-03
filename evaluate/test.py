from evaluate_utils import XMLTokenizer
from evaluate_utils import sentence_bleu, remove_xml_attributes
from nltk.translate.bleu_score import SmoothingFunction
from evaluate_utils import compute_bleu_rouge

tokenizer = XMLTokenizer(stemm=False)
pred = """Der böhmische Kanzler, der in der Region von Böhmen eine entscheidende Rolle gespielt hat, war ein wichtiger Akteur während der Religionskriege des 16. Jahrhunderts. Das Referenz zu "von Blauwen" könnte sich auf den Adelsgeschlecht von Blau oder ähnlichen Familien beziehen, die in dieser Zeit Einfluss in Böhmen hatten."""
ref = """Heinrich von Plauen, Burggraf von Meißen und böhmischer Kanzler."""
pred = """<persName ref="p18988" cert="high">Cyprianus</persName>, De Lapsis, 15, 1; CSEL 3, 56–57."""
ref = """Vgl. bes. <persName ref="p18988" cert="high">Cyprian</persName>, Epist. 63, 10, 2-11, 1 und 14, 1-3 (CChr III C 402f. 408f)."""
pred = """Gemeint ist <persName ref="p8051" cert="high">Anna Bullinger, geb. Adlischwyler</persName>."""
ref = """<persName ref="p8051" cert="high">Anna, geb. Adlischwyler</persName>."""
print(remove_xml_attributes(ref))
print(tokenizer.tokenize(pred))
print(compute_bleu_rouge((pred, ref)))
pred = tokenizer.tokenize(pred)
ref = tokenizer.tokenize(ref)
chencherry = SmoothingFunction()
print(sentence_bleu([ref], pred, smoothing_function=chencherry.method3))
