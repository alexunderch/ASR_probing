from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
from nltk.tree import Tree, ParentedTree
import nltk
import numpy as np

text = (
  'Pusheen and Smitha walked along the beach. '
  'Pusheen wanted to surf, but fell off the surfboard. '
  'There is a big shit.')
output = nlp.annotate(text, properties={
  'annotators': 'tokenize,ssplit,pos,depparse,parse',
  'outputFormat': 'json'
  })

tree = Tree.fromstring(output['sentences'][1]['parse'])
tree.pretty_print()
print(str(tree)[5:])

for ind, leaf in enumerate(tree.leaves()):
    print(leaf)
    # tree_location = tree.leaf_treeposition(ind)[:-1]
    # print(tree_location)
    # print(tree[tree_location])
    # for lleaf in tree[tree_location].leaves():
    #     print(lleaf, "de")


print(tree[(0, 0, 0, 0)], tree[(0, 0, 0)], tree[(0, 0)], tree[0])
for subtree in tree.subtrees():
    print(subtree.label(), subtree.height())

print(tree.productions())

def my_traverse(productions: list):
    productions = [str(p).strip() for p in productions]
    starter = productions[0].split("->")[-1].strip()
    level = np.array(productions)[list(p.startswith(starter + " ") for p in productions)]
    for l in level:
        print(l.split("->")[-1].split())
_ = my_traverse(tree.productions())



############################################################################################################################

import queue
from datasets import Dataset
from pycorenlp import StanfordCoreNLP
import nltk.tree as tree
import numpy as np
from typing import List
from collections import Counter


class TopConstituency(object):
    def __init__(self, dataset: Dataset, data_column: str = "text") -> None:
        self.dataset = dataset
        self.data_col = data_column
        self.stats = Counter()
        self.nlp_annotator = StanfordCoreNLP('http://localhost:9000')
        
    
    def annotate_dataset(self, key: str):
        """tokenize, ner, pos, depparse, parse"""

        def annotate_batch(batch, key: str):
            output = nlp.annotate(batch[self.data_col], properties = {
                'annotators': 'tokenize, ssplit, pos, depparse, parse, ner',
                'outputFormat': 'json'
            })
            batch[key] = output[key] ###?
            return batch
        
        self.dataset = self.dataset.map(annotate_batch, fn_kwargs = {"key": key})

    def found_the_most_popular(self, thr: int) -> List:
        """Filtering the Counter"""
        return list(x for x, count in self.stats.items() if count >= thr)  + ["other"]

    def top_constituency(self, top_k: int = None): 

        def search_levelwise(_tree: tree.Tree, depth: int = top_k): pass
        
        def batched_search(batch, top_k: int):
            tmp_tree = tree.Tree.fromstring(batch["parse"])
            """BFS"""
            return batch
            
        
             

