import argparse
import sys
from collections import Counter

description='Extract anchor words from tagged sentences'
argparser = argparse.ArgumentParser(description)
argparser.add_argument('tagged_file', type=str, help='file of tagged sentences')
argparser.add_argument('--cap', action='store_true', help='also consider capitalized words')
argparser.add_argument('--all', action='store_true', help='use all tags, not just open-class')
argparser.add_argument('--n', type=int, default=int(1e10), help='just use the n most frequent words')
args = argparser.parse_args()

OPEN_CLASS = {'NOUN':True, 'VERB':True, 'ADJ':True, 'ADV':True}
words = {}
tags = {}

lines = open(args.tagged_file).readlines()
for line in lines:
    toks = line.split()
    if len(toks) == 0:
        continue
    word = toks[0]
    tag = toks[1]
    if (not args.cap) and word[0].isupper():
        continue
    if (not args.all) and (not tag in OPEN_CLASS):
        continue

    if not word in tags:
        tags[word] = Counter()
    tags[word][tag] += 1

    if not tag in words:
        words[tag] = Counter()
    words[tag][word] += 1

anchor_words = {} 
for tag in words:
    for word in words[tag]:
        if len(tags[word]) == 1:
            if not tag in anchor_words:
                anchor_words[tag] = {}
            anchor_words[tag][word] = words[tag][word]

sorted_words = {}
for tag in anchor_words:
    sorted_words[tag] = sorted(anchor_words[tag].items(), key=lambda x: x[1], reverse=True)

print 'Have {} tags with anchor words:'.format(len(anchor_words))
for tag in anchor_words:
    print '{} has {} anchor words, such as {}'.format(tag, len(anchor_words[tag]), sorted_words[tag][0:min(7, len(sorted_words[tag]))])

with open('anchor_words.txt', 'wb') as outf:
    for tag in sorted_words:
        for word, count in sorted_words[tag][:args.n]:
            print >> outf, word, tag, count





