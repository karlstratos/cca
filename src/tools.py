import sys
import os
import gc
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
                    say('/==%dm===/' % status, False)
                    status += 1
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

                     

                    