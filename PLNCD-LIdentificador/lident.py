#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Jerónimo Carranza Carranza

Identificador de idioma basado en N-grams (Tri-gramas)
'N-Gram-Based Text Categorization (1994) - William B. Cavnar , John M. Trenkle'
Adaptado de blog.alejandronolla.com
Corpus de lenguas: European Parliament Proceedings Parallel Corpus 1996-2011. v7. http://www.statmt.org/europarl/

'''


import glob
import operator
import os

from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams

langdata_path = './ldata/'
tokenizer = RegexpTokenizer("[a-zA-ZñÑ'`áéíóúüèî]+")

languages_statistics = {}


# ----------------------------------------------------------------------
def generate_ngrams(tokens):
    generated_ngrams = []
    for token in tokens:
        for x in range(3, 4):
            xngrams = ngrams(token, x, pad_left=False, pad_right=False)
            for xngram in xngrams:
                ngram = ''.join(xngram)
                generated_ngrams.append(ngram)
    return generated_ngrams


# ----------------------------------------------------------------------
def calculate_ngram_occurrences(text):
    ngrams_statistics = {}
    tokens = tokenizer.tokenize(text.lower())
    ngrams = generate_ngrams(tokens)
    for ngram in ngrams:
        if not ngram in ngrams_statistics:
            ngrams_statistics.update({ngram: 1})
        else:
            ngram_occurrences = ngrams_statistics[ngram]
            ngrams_statistics.update({ngram: ngram_occurrences + 1})
    ngrams_statistics_sorted = sorted(ngrams_statistics.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)[0:300]
    return ngrams_statistics_sorted


# ----------------------------------------------------------------------
def generate_ngram_from_file(file_path, output_filename):
    raw_text = open(file_path, mode='r').read()
    output_filenamepath = os.path.join(langdata_path, output_filename)
    profile_ngrams_sorted = calculate_ngram_occurrences(raw_text)
    fd = open(output_filenamepath, mode='w')
    for ngram in profile_ngrams_sorted:
        fd.write('%s\t%s\n' % (ngram[0], ngram[1]))
    fd.close()


# ----------------------------------------------------------------------
def load_profiles():
    languages_files = glob.glob('%s*.dat' % langdata_path)
    for language_file in languages_files:
        filename = os.path.basename(language_file)
        language = os.path.splitext(filename)[0]
        ngram_statistics = open(language_file, mode='r').readlines()
        ngram_statistics = [n.strip('\n').split('\t') for n in ngram_statistics]
        languages_statistics.update({language: ngram_statistics})


# ----------------------------------------------------------------------
def compare_profiles(category_profile, document_profile):
    document_distance = 0
    category_ngrams_sorted = [ngram[0] for ngram in category_profile]
    document_ngrams_sorted = [ngram[0] for ngram in document_profile]
    maximum_out_of_place_value = max(len(document_ngrams_sorted) + 1, len(category_ngrams_sorted) + 1)
    for ngram in document_ngrams_sorted:
        document_index = document_ngrams_sorted.index(ngram)
        if ngram in category_ngrams_sorted:
            category_profile_index = category_ngrams_sorted.index(ngram)
        else:
            category_profile_index = maximum_out_of_place_value
        distance = abs(category_profile_index - document_index)
        document_distance += distance
    return document_distance


# ----------------------------------------------------------------------
def what_language(text):
    languages_ratios = {}
    load_profiles()
    text_ngram_statistics = calculate_ngram_occurrences(text)
    for language, ngrams_statistics in languages_statistics.items():
        distance = compare_profiles(ngrams_statistics, text_ngram_statistics)
        languages_ratios.update({language: distance})
    nearest_language = min(languages_ratios, key=languages_ratios.get)
    return nearest_language


# ----------------------------------------------------------------------
def train():
    generate_ngram_from_file('./corpus/mini_europarl-v7.es-en.en', 'EN_1000.dat')
    generate_ngram_from_file('./corpus/mini_europarl-v7.es-en.es', 'ES_1000.dat')
    generate_ngram_from_file('./corpus/mini_europarl-v7.fr-en.fr', 'FR_1000.dat')
    generate_ngram_from_file('./corpus/mini_europarl-v7.it-en.it', 'IT_1000.dat')
    generate_ngram_from_file('./corpus/mini_europarl-v7.pt-en.pt', 'PT_1000.dat')

    generate_ngram_from_file('./corpus/100k_europarl-v7.es-en.en', 'EN_100k.dat')
    generate_ngram_from_file('./corpus/100k_europarl-v7.es-en.es', 'ES_100k.dat')
    generate_ngram_from_file('./corpus/100k_europarl-v7.fr-en.fr', 'FR_100k.dat')
    generate_ngram_from_file('./corpus/100k_europarl-v7.it-en.it', 'IT_100k.dat')
    generate_ngram_from_file('./corpus/100k_europarl-v7.pt-en.pt', 'PT_100k.dat')


# ----------------------------------------------------------------------
def test():

    test_texts = ['Merci beaucoup','Muchas gracias','Grazie mille','Thank you very much','Muito obrigado']

    for tt in test_texts:
        print(tt, '\t', what_language(tt))


    print('\nMatriz de divergencia entre perfiles:\n')
    for l1, stat1 in languages_statistics.items():
        row = [l1]
        for l2, stat2 in languages_statistics.items():
            row += [round(compare_profiles(stat1, stat2)/1000,1)]
        print(row)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    test()
