from dataclasses import replace
from collections import deque
import queue
from re import search
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
from nltk.tree import Tree, ParentedTree
import nltk
from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
from collections import Counter

#https://stanfordnlp.github.io/CoreNLP/download.html

#################################################################################################Helpers
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

def find_path_from_root(ptree, text):
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
    # ptree.pretty_print()
    paths = []
    for leave in ptree.leaves(): paths.append("_".join(find_path_from_root(ptree, leave)[:thr]) + "_.")
    return paths


def get_constituency(batch, data_column: str, key: str):
    output = nlp.annotate(batch[data_column], properties = {'annotators': 'tokenize, ner, pos, depparse, parse',
                                                            'outputFormat': 'json' })
    batch['constituencies'] = list()
    for out in output['sentences']:
        batch['constituencies'].extend(parse_paths(out[key]) + ['OTHER'])
    
    batch['constituencies'] = list(set(batch['constituencies']))
    return batch


def collect_statistics(batch, labels: list):
    """just flatten the sentences"""
    labels.extend(batch["constituencies"])
    return batch


def label_map(batch, labels: dict):
    """labeling dataset according to given labeling: from the most frequent to the least"""
    batch["top_constituency"] = "OTHER"
    for label in labels.keys():
        if label in batch["constituencies"]: 
            print(label,  batch["constituencies"])
            batch["top_constituency"] = label
            break
    return batch
#################################################################################################

def prepare_dataset(dname: str, text_col: str):
    d = load_dataset(dname, split = "train")
    d = d.map(get_constituency, fn_kwargs = {"data_column": text_col, "key": "parse"})
    d.save_to_disk(f"parsed_{dname}_large")

_ = prepare_dataset("timit_asr", "text")

def prepare_labels(dname_str:str):
    data = load_from_disk(dname_str)
    labels = list()

    data = data.map(collect_statistics, fn_kwargs = {"labels": labels})
    stats = Counter(labels)

    def filter_(stats: Counter, low: int, high: int):
        from string import punctuation
        punct = [f"_{c}_" for c in punctuation]
        def fpunct(expr: str):
            for p in punct:
                if p in expr: return False
            return True

        return {k: count for k, count in stats.items() 
                if count >= low and count <= high and fpunct(k)}
        
    stats = filter_(stats, 10, 1000)
    stats = dict(sorted(stats.items(), key=lambda item: -item[1]))
    print(stats, len(stats))
    return_ = {k: ind for ind, k in enumerate(list(stats.keys()))}
    _ = return_.update({"OTHER": len(stats)})
    return return_


def label_dataset(dname_str:str, labels: dict):
    data = load_from_disk(dname_str)
    return data.map(label_map, fn_kwargs = {"labels": labels})

d = label_dataset("parsed_timit_asr_large", prepare_labels("parsed_timit_asr_large"))
d.save_to_disk("parsed_timit_asr_large")
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
fig = plt.figure(figsize=(15,4))
ax = plt.gca()
counts, _, patches = ax.hist(list(d['top_constituency']), bins=len(set(d['top_constituency'])),edgecolor='r')
for count, patch in zip(counts,patches):
    ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()))
plt.show()

#parsing sexpr
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
