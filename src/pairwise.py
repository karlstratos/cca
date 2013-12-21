import argparse
import numpy
from io import say
from io import set_quiet
from pca import pca_svd

def pairwise_classify(embedding_file, anchor_words, no_balance=False):
    wdic, tdic = read_anchors(anchor_words)
    rep = read_rep(embedding_file, tdic) # only read embeddings that we need
    acc_all = 0.
    check = {}
    for tag1 in wdic:
        for tag2 in wdic:            
            if tag1 != tag2 and not (tag2, tag1) in check:
                if no_balance:
                    acc_all += classify(tag1, tag2, wdic[tag1], wdic[tag2], rep, no_balance)
                else: 
                    mi = min(len(wdic[tag1]), len(wdic[tag2])) # equal number of words for each side                    
                    acc_all += classify(tag1, tag2, wdic[tag1][:mi], wdic[tag2][:mi], rep, no_balance)
                check[(tag1, tag2)] = True
    acc_all /= len(check)
    say('overall acc: {}'.format(acc_all))
    return acc_all
                                        
def classify(tag1, tag2, wdic1, wdic2, rep, no_balance):
    true_tag = {}
    num1 = len(wdic1)
    num2 = len(wdic2)
    wind = {}
    iwrd = {}
    wnum = 0

    for word in wdic1:
        true_tag[word] = tag1
        wind[word] = wnum
        iwrd[wnum] = word
        wnum += 1

    for word in wdic2:
        true_tag[word] = tag2
        wind[word] = wnum
        iwrd[wnum] = word
        wnum += 1
    
    A = numpy.zeros((wnum, len(rep[rep.keys()[0]])))
    for i in range(wnum):
        A[i,:] = rep[iwrd[i]] if iwrd[i] in rep else rep['<?>']
                
    pca_trans, _, _ = pca_svd(A) 
    pca1_vals = pca_trans[:,0]
    
    best_correct = 0
    best_acc = float('-inf')
    
    if no_balance:
        myrange = range(len(pca_trans))
    else:
        middle = len(pca_trans)/2
        surrounding = min(middle, 100) 
        myrange = range(middle - surrounding, middle + surrounding)
    
    for j in myrange:
        correct = 0
        indices = sorted(range(len(pca1_vals)), key=lambda i: pca1_vals[i])
        for word_i in indices:
            word = iwrd[word_i]
            pca1_val = pca1_vals[word_i]
            pred_tag = tag1 if pca1_val > pca1_vals[j] else tag2
            if word in true_tag:
                if pred_tag == true_tag[word]:
                    correct += 1
        
        acc = float(correct)/len(pca1_vals) * 100
        
        reverse = False
        if acc < 50:
            correct = len(indices)-correct
            acc = float(correct)/len(pca1_vals) * 100
            reverse = True

        for word_i in indices:
            word = iwrd[word_i]
            pca1_val = pca1_vals[word_i]
            if not reverse:
                pred_tag = tag1 if pca1_val > 0 else tag2
            else:
                pred_tag = tag2 if pca1_val > 0 else tag1    
        
        if acc > best_acc:
            best_correct = correct
            best_acc = acc 
    
    say('acc: {}% ({} / {})\t'.format(best_acc, best_correct, len(indices)), False)        
    say('{} {}, {} {}'.format(num1, tag1, num2, tag2))
    return best_acc

def read_rep(embedding_file, tdic):
    rep = {}
    with open(embedding_file) as f:
        for line in f:    
            toks = line.split()
            word = toks[1]
            if word in tdic or word == '<?>':
                rep[word] = map(lambda x: float(x), toks[2:])
    return rep

def read_anchors(anchor_words):
    wdic = {}
    tdic = {}
    with open(anchor_words) as f:
        for line in f:
            toks = line.split()
            if len(toks) > 0:
                word = toks[0]
                tag = toks[1]
                if not tag in wdic:
                    wdic[tag] = [] 
                wdic[tag].append(word) # words assumed ordered in decreasing frequency
                tdic[word] = tag
    return wdic, tdic

if __name__=='__main__':
    argparser = argparse.ArgumentParser('Perform binary classification between every label pair')
    argparser.add_argument('embedding_file', type=str, help='file of embeddings')
    argparser.add_argument('anchor_words',   type=str, help='list of anchor words to classify')
    argparser.add_argument('--no_balance', action='store_true', help='don\'t balance label pairs')
    argparser.add_argument('--quiet', action='store_true', help='quiet mode')
    args = argparser.parse_args()

    set_quiet(args.quiet)
    pairwise_classify(args.embedding_file, args.anchor_words, args.no_balance)