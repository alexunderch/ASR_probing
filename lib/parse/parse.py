from dataclasses import replace
from collections import deque
import queue
from re import search
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
from nltk.tree import Tree, ParentedTree
import nltk

import numpy as np
#https://stanfordnlp.github.io/CoreNLP/download.html
text = (
  'Pusheen and Smitha walked along the beach. '
  'Pusheen wanted to surf, but fell off the surfboard. '
  'There is a big shit.')
output = nlp.annotate(text, properties={
  'annotators': 'tokenize,ssplit,pos,depparse,parse',
  'outputFormat': 'json'
  })

def tree_to_dict(tree):
    return {tree.label(): [tree_to_dict(t)  if isinstance(t, nltk.Tree) else t for t in tree]}


def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def get_labels_from_lca(ptree, lca_len, location):
    labels = []
    for i in range(lca_len, len(location)):
        labels.append(ptree[location[:i]].label())
    return labels

def findPath(ptree, text):
    leaf_values = ptree.leaves()
    leaf_index1 = leaf_values.index(text)
    location = ptree.leaf_treeposition(leaf_index1)
    lca_len = 0
    labels1 = get_labels_from_lca(ptree, lca_len, location)
    result = labels1[1:]
    return result

def find_path_between(ptree, text1, text2):
    leaf_values = ptree.leaves()
    leaf_index1 = leaf_values.index(text1)
    leaf_index2 = leaf_values.index(text2)

    location1 = ptree.leaf_treeposition(leaf_index1)
    location2 = ptree.leaf_treeposition(leaf_index2)

    #find length of least common ancestor (lca)
    lca_len = get_lca_length(location1, location2)

    #find path from the node1 to lca

    labels1 = get_labels_from_lca(ptree, lca_len, location1)
    #ignore the first element, because it will be counted in the second part of the path
    result = labels1[1:]
    #inverse, because we want to go from the node to least common ancestor
    result = result[::-1]

    #add path from lca to node2
    result = result + get_labels_from_lca(ptree, lca_len, location2)
    return result

def parse_paths(text, thr = 4):
    ptree = ParentedTree.fromstring(text)
    ptree.pretty_print()
    paths = []
    for leave in ptree.leaves(): paths.append("_".join(findPath(ptree, leave)[:thr]) + "_")
    return paths

print(parse_paths(output['sentences'][0]['parse']))

def get_constituency(batch, data_column: str, key: str):
    output = nlp.annotate(batch[data_column], properties = {'annotators': 'tokenize, ner, pos, depparse, parse',
                                                            'outputFormat': 'json' })
    batch['constituencies'] = list()
    for out in output['sentences']:
        batch['constituencies'].extend(parse_paths(out[key]))
    return batch



# _paths(tree)
####https://gist.github.com/pib/240957
from string import whitespace

atom_end = set('()"\'') | set(whitespace)

def parse(sexp):
    stack, i, length = [[]], 0, len(sexp)
    while i < length:
        c = sexp[i]

        # print (c , stack) 
        reading = type(stack[-1])
        if reading == list:
            if   c == '(': stack.append([])
            elif c == ')': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '"': stack.append('')
            elif c == "'": stack.append([('quote',)])
            elif c in whitespace: pass
            else: stack.append((c,))
        elif reading == str:
            if   c == '"': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '\\': 
                i += 1
                stack[-1] += sexp[i]
            else: stack[-1] += c
        elif reading == tuple:
            if c in atom_end:
                atom = stack.pop()
                if atom[0][0].isdigit(): stack[-1].append(eval(atom[0]))
                else: stack[-1].append(atom[0])
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
                continue
            else:  stack[-1] = ((stack[-1][0] + c),)
        i += 1
    return stack.pop()

a = output['sentences'][1]['parse']
# print(repr(a.replace("(", "").replace(")", "")))
r =  parse(a)
# print("parsed", r)

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
            

