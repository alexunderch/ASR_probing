from datasets import load_dataset, load_from_disk
import numpy as np
import os
from corpy.udpipe import Model, pprint
from conllu import parse
def download_english_model() -> int:
    return os.system("wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe?sequence=17&isAllowed=y")

parser_model = Model("./english-ewt-ud-2.5-191206.udpipe")

def preprocess_batch(batch, data_col: str):
    sent = list(parser_model.process(batch[data_col]))[0]
    feats = str()
    for word in sent.words: feats += str(word.feats)
    
    if "Number=Sing" in feats: batch['obj_number'] = "single"
    elif "Number=Plur" in feats:  batch['obj_number'] = "plural"
    else: batch['obj_number'] = "other"

    if "Tense=Past|VerbForm=Fin" in feats: batch['tense'] = "past"
    elif "Tense=Pres|VerbForm=Fin" in feats: batch['tense'] = "present"
    else: batch['tense'] = "other"
        
    return batch

def test():
    sents = list(parser_model.process("Don't ask me to carry an oily rag like that."))
    obj_number, tense = [], []
    for sent in sents:
        feats = str()
        for word in sent.words:
            feats += str(word.feats) + '/n'
        
        print(feats)
        if "Number=Sing" in feats: obj_number.append("single")
        elif "Number=Plur" in feats: obj_number.append("plural")
        else: obj_number.append("other")

        if "Tense=Past|VerbForm=Fin" in feats: tense.append("past")
        elif "Tense=Pres|VerbForm=Fin" in feats: tense.append("present")
        else: tense.append("other")
        
    print(obj_number)
    print(tense)
    

def main(dname: str, from_disk: bool = False):
    dataset = load_dataset(dname, split = "train") if not from_disk else load_from_disk(dname)

    dataset = dataset.map(preprocess_batch, fn_kwargs = {"data_col": "text"})
    dataset.save_to_disk(dname + "__tense__obj_number")

if __name__ == "__main__": test()
