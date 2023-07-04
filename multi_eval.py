import os
import sys
import csv
from transformers import BartTokenizer 
import json
import argparse
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
import nltk.translate.nist_score as nist_score
from nlgeval import NLGEval
import collections

nlgeval = NLGEval(metrics_to_omit=['CIDEr','ROUGE_L','METEOR','EmbeddingAverageCosineSimilarity','VectorExtremaCosineSimilarity','GreedyMatchingScore','SkipThoughtCS']) 

cc = SmoothingFunction()

def clean_tokenize_sentence(data):
    data = data.lower().strip()#.replace(" \' ", "\'")
    spacy_token = tokenizer.tokenize(data)
    input(spacy_token)
    # spacy_token = nlp(data)
    
    if len(spacy_token)>0 and spacy_token[-1].text == 'eos':
        spacy_token = spacy_token[:-2]
    if len(spacy_token)>0 and spacy_token[0].text == '_':
        spacy_token = spacy_token[2:]

    if len(spacy_token)==0:
        return ['.']
    return [(token.text) for token in spacy_token]

def calculate_max_bleu(list_references, list_hypothesis, weights):
    sum_bleu = 0.0
    for i,d in enumerate(list_references):
        references_items = list_references[i]
        hypothesis = list_hypothesis[i]
        bleu_score_sentence = []
        for reference in references_items:
            bleu_score_sentence.append(sentence_bleu([reference], hypothesis, weights, smoothing_function=cc.method1))
        sum_bleu += max(bleu_score_sentence)
    mean_bleu = sum_bleu / len(list_hypothesis)   

    return mean_bleu

def convert_tostring_lists(list_references, list_hypothesis):
    list_string_references, list_string_hypothesis = [], []
    for hypothesis_itemlist in list_hypothesis:
        list_string_hypothesis.append(' '.join(hypothesis_itemlist))

    for references_list in list_references:
        list_string_references.append([' '.join(individual_ref) for individual_ref in references_list])

    ##convert references to n separate lists of referenes
    num_responses = len(list_string_references[0])
    mod_reference_list = [[] for i in range(num_responses)]
    for i, item in enumerate(list_string_references):
        for j, ref in enumerate(item):
            mod_reference_list[j].append(ref)

    return mod_reference_list, list_string_hypothesis

def print_metrics_dict(metrics_dict):
    for metric in metrics_dict.keys():
        print(metric + ';\t' + str(metrics_dict[metric] * 100))

def get_all_metrics(list_references, list_hypothesis):

    list_string_references, list_string_hypothesis = convert_tostring_lists(list_references, list_hypothesis)
    metrics_dict = nlgeval.compute_metrics(list_string_references, list_string_hypothesis)
    print_metrics_dict(metrics_dict)

def calc_diversity(hyp):
    # based on Yizhe Zhang's code
    tokens = [0.0, 0.0]
    types = [collections.defaultdict(int), collections.defaultdict(int)]
    for line in hyp:
        for n in range(2):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys()) / tokens[0]
    div2 = len(types[1].keys()) / tokens[1]
    return [div1, div2]

if __name__ == '__main__':
  
    data_path = sys.argv[1]
      
    os.system('grep ^H {} | LC_ALL=C sort -V | cut -f3- > hyp.txt'.format(data_path))
    os.system('grep ^T {} | LC_ALL=C sort -V | cut -f2- > ref.txt'.format(data_path))
    
    list_references = []
    list_hypothesis = []

    with open('hyp.txt','r') as hyp, open('dd_dataset/test.refs','r') as refs:
        hyp_lines = hyp.readlines()
        refs_lines = refs.readlines()

        for hyp_line, refs_line in zip(hyp_lines, refs_lines):
            list_hypothesis.append(hyp_line.strip().split(' '))
            refs_line = refs_line.strip().split('\t')
            list_references.append([ref.strip().split(' ') for ref in refs_line])
        
    distinct = [round(x * 100, 2) for x in calc_diversity(list_hypothesis)]
    print('dist1:',distinct[0])
    print('dist2:',distinct[1])
    get_all_metrics(list_references, list_hypothesis)