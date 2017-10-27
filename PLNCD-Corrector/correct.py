#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Jerónimo Carranza Carranza

Corrector simple basado en el Corpus de Referencia del Español Actual (CREA)
y sus frecuencias (> 1)
http://corpus.rae.es/lfrecuencias.html

'''

import csv

alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóúüñ'


def read_CREA(file='CREA_total.TXT', min_frec=0):
    frecs = {}
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader, None)
        for r in reader:
            frec = int(r[2].replace(",", ""))
            if frec > min_frec:
                frecs[r[1]] = frec
    return frecs


NWORDS = read_CREA(min_frec=2)


def known(words): return set(w for w in words if w in NWORDS)  

   
def edits1(word):  
   s = [(word[:i], word[i:]) for i in range(len(word) + 1)]  
   deletes    = [a + b[1:] for a, b in s if b]  
   transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]  
   replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]  
   inserts    = [a + c + b     for a, b in s for c in alphabet]  
   return set(deletes + transposes + replaces + inserts)  

  
def known_edits2(word):  
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)  

  
def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    cand = sorted(candidates, key=NWORDS.get, reverse=True)[0:10]
    return cand


def test():
    test_words = ['urogayo','pyromano','bevía','kaminar','cigueña',
                  'zigota','alacena','atronar','alunizar','abrebia']

    for tw in test_words:
        print(tw, ' -> ', correct(tw))


if __name__ == '__main__':
    test()
