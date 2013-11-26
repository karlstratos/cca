import sys
import os
import gc
import numpy
from collections import Counter
from collections import deque

_buffer_ = '<*>'
_rare_ = '<?>'
_status_unit_ = 1000000


def say(string, newline=True):
    if newline:
        print string
        sys.stdout.flush()
    else:
        print string,
        sys.stdout.flush()


def count_ngrams(corpus, n_vals=False):
    assert(os.path.isfile(corpus))    
    if n_vals == False:
        answer = raw_input('Type in the values of n (e.g., \"1 3\"): ')        
        n_vals = [int(n) for n in answer.split()]
    
    say('\nCounting... ', False)
    num_tok = 0
    status = 0
    ngrams = [Counter() for n in n_vals]                                      
    queues = [deque([_buffer_ for _ in range(n-1)], n) for n in n_vals]
    with open(corpus) as f: 
        for line in f:
            toks = line.split()
            for tok in toks:
                num_tok += 1
                if num_tok % _status_unit_ == 0:
                    status += 1
                    say('/==%dm===/' % status, False)
                for i in range(len(n_vals)):
                    queues[i].append(tok)
                    ngrams[i][tuple(queues[i])] += 1
                 
    for i in range(len(n_vals)):
        for _ in range(n_vals[i]-1):
            queues[i].append(_buffer_)
            ngrams[i][tuple(queues[i])] += 1

    say('\nTotal {} tokens'.format(num_tok))
    files = [os.path.dirname(corpus)+'/'+os.path.splitext(os.path.basename(corpus))[0]+'.'+str(n)+'grams' for n in n_vals]        
    for i in range(len(n_vals)):
        say('Sorting {} {}grams and writing to: {}'.format(len(ngrams[i]), n_vals[i], files[i]))
        sorted_ngrams = sorted(ngrams[i].items(), key=lambda x: x[1], reverse=True)
        with open(files[i], 'wb') as outf:
            for ngram, count in sorted_ngrams:
                for tok in ngram:
                    print >> outf, tok,
                print >> outf, count
            ngrams[i] = {}; del sorted_ngrams; gc.collect() # try to free some memory


def cutoff_rare(ngrams, cutoff, unigrams):
    assert(os.path.isfile(unigrams))    
    say('Reading unigrams')
    rare = Counter()
    num_unigrams = 0 
    with open(unigrams) as f:
        for line in f:
            num_unigrams += 1
            toks = line.split()
            word = toks[0]
            count = int(toks[1])
            if count <= cutoff:
                rare[word] = count
    
    say('Only {} out of {} words appear > {}, will replace other {} with \"<?>\" token in {}'.\
                                    format(num_unigrams - len(rare), num_unigrams, cutoff, len(rare), ngrams))

    new_ngrams = Counter()    
    n = 0
    with open(ngrams) as f:
        for line in f:
            toks = line.split()
            ngram = toks[:-1]
            n = len(ngram)
            count = int(toks[-1])
            new_ngram = []
            for gram in ngram:
                if gram in rare:
                    new_ngram.append(_rare_)
                else:
                    new_ngram.append(gram)
            new_ngrams[tuple(new_ngram)] += count
            
    outfname = ngrams + '.cutoff' + str(cutoff)
    say('Sorting {} {}grams and writing to: {}'.format(len(new_ngrams), n, outfname))
    sorted_ngrams = sorted(new_ngrams.items(), key=lambda x: x[1], reverse=True)
    with open(outfname, 'wb') as outf:
        for ngram, count in sorted_ngrams:
            for gram in ngram:
                print >> outf, gram,
            print >> outf, count
            

def phi(token, rel_position):
    if rel_position > 0:
        position_marker = '<+'+str(rel_position)+'>'
    elif rel_position < 0:
        position_marker = '<'+str(rel_position)+'>'
    else:
        position_marker = ''
    feat = token+position_marker
    holder = {feat : True}
    return holder


def extract_views(ngrams):
    outfname = ngrams + '.featurized'
    say('Writing the featurized file to: ' + outfname)
    with open(outfname, 'wb') as outf:
        with open(ngrams) as f:
            for line in f:
                toks = line.split()
                ngram = toks[:-1]
                count = int(toks[-1])
                center = len(ngram) / 2 # position of the current word
                print >> outf, count,

                # definition of view 1
                view1_holder = phi(ngram[center], 0)
                for view1f in view1_holder:
                    print >> outf, view1f,
                
                print >> outf, '|:|',
                
                # definition of view 2
                for i in range(len(ngram)):
                    if i != center:
                        view2_holder = phi(ngram[i], i-center)
                        for view2f in view2_holder:
                            print >> outf, view2f,

                print >> outf


def pca(A):
    """ performs principal components analysis 
        (PCA) on the n-by-p data matrix A
        Rows of A correspond to observations, columns to variables. 
    """
    M = (A-numpy.mean(A.T,axis=1)).T 
    [evals,evecs] = numpy.linalg.eig(numpy.cov(M)) 
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]
    pca_vals = numpy.dot(evecs.T,M) # projection of the data in the new space
    return evecs, pca_vals, evals
                     

                    
def nv_classify(A, nv_list):
    true_tag = {}
    v_num = 0
    n_num = 0
    wdic = {}
    wdici = {}
    wnum = 0
    lines = open(nv_list).readlines()
    for line in lines:
        word, tag = line.split()
        true_tag[word] = tag
        if tag == 'v':
            v_num += 1
        elif tag == 'n':
            n_num += 1
        else:
            print 'v/n format is wrong'
            exit(-1)
        wdic[word] = wnum
        wdici[wnum] = word
        wnum += 1
    
    rep = {}
    dim = 0
    lines = open(A).readlines()
    for line in lines:
        toks = line.split()
        word = toks[1]
        if word in true_tag or word == '<?>':
            if word == '<?>':
                wdic[word] = wnum
                wdici[wnum] = word
                wnum += 1
            rep[wdic[word]] = map(lambda x: float(x), toks[2:])
            dim = len(rep[wdic[word]])
    
    Amat = numpy.zeros((len(rep), dim))
    for word_i in rep:
        if word_i < Amat.shape[0]:
            try:
                Amat[word_i,:] = rep[word_i]
            except:
                Amat[word_i,:] = rep[wdic['<?>']]
                
    _, pca_vals, _ = pca(Amat) 
    
    pca1_vals = pca_vals[0,:]
    
    correct = 0
    indices = sorted(range(len(pca1_vals)), key=lambda i: pca1_vals[i])
    for word_i in indices:
        word = wdici[word_i]
        pca1_val = pca1_vals[word_i]
        pred_tag = 'v' if pca1_val > 0 else 'n'
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
        word = wdici[word_i]
        pca1_val = pca1_vals[word_i]
        if not reverse:
            pred_tag = 'v' if pca1_val > 0 else 'n'
        else:
            pred_tag = 'n' if pca1_val > 0 else 'v'
    
    say('acc: {}% ({} / {})'.format(acc, correct, len(indices)))        
    say('{} verbs, {} nouns'.format(v_num, n_num))


