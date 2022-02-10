from lib.linguistic_utils import BertProbingDataset
import os
def main():
    os.system("wget https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/past_present.txt")
    dtst = BertProbingDataset('bert', './obj_number.txt', "obj_number_set", 'obj_number')
    _ = dtst.make_dataset()

    os.system("wget https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/obj_number.txt")
    dtst = BertProbingDataset('bert', './past_present.txt', "tense_set", 'tense')
    _ = dtst.make_dataset()

    del dtst


if __name__ == "__main__": main()