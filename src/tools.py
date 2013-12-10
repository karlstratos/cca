import sys
import os
import gc
import numpy
from collections import Counter
from collections import deque
from pca import pca_svd

# shared variables 
_quiet_ = False
_buffer_ = '<*>'
_rare_ = '<?>'
_status_unit_ = 1000000


def set_quiet(quiet):
    global _quiet_
    _quiet_ = quiet


def say(string, newline=True):
    if not _quiet_:        
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
    
    say('Counting... ', False)
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


def cutoff_rare(ngrams, cutoff, unigrams, given_myvocab):
    assert(unigrams and os.path.isfile(unigrams)) 

    if(given_myvocab):
        myvocab = {}
        myvocab_hit = {}
        with open(given_myvocab) as f:
            for line in f:
                toks = line.split()
                if len(toks) == 0:
                    continue
                myvocab[toks[0]] = True
    
    say('Reading unigrams')
    vocab = {}
    num_unigrams = 0 
    with open(unigrams) as f:
        for line in f:
            num_unigrams += 1
            toks = line.split()
            if len(toks) != 2:
                continue
            word = toks[0]
            count = int(toks[1])
            
            if count > cutoff:
                vocab[word] = count

            if given_myvocab and word in myvocab:
                vocab[word] = count
                myvocab_hit[word] = True

    say('Will keep {} out of {} words'.format(len(vocab), num_unigrams))
    if given_myvocab:
        say('\t- They include {} out of {} in my vocab'.format(len(myvocab_hit), len(myvocab)))

    vocab['_START_'] = True # for google n-grams 
    vocab['_END_'] = True    
    
    ans = raw_input('Do you want to proceed with the setting? [Y/N] ')
    if ans == 'N' or ans == 'n':
        exit(0)

    new_ngrams = Counter()    
    n = 0
    temp = {}
    with open(ngrams) as f:
        for line in f:
            toks = line.split()
            ngram = toks[:-1]
            n = len(ngram)
            count = int(toks[-1])
            new_ngram = []
            for gram in ngram:
                this_tok = gram if gram in vocab else _rare_
                new_ngram.append(this_tok)
                temp[this_tok] = True
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


def perform_pca(embedding_file, pca_dim):
    freqs, words, A = read_embeddings(embedding_file)
    
    say('performing PCA to reduce dimensions from {} to {}'.format(A.shape[1], pca_dim))            

    pca_trans, _, _ = pca_svd(A) 
    A_pca = pca_trans[:,:pca_dim]
    
    write_embeddings(freqs, words, A_pca, embedding_file + '.pca' + str(pca_dim))

def read_embeddings(embedding_file):
    freqs = {}
    words = {}
    rep = {}
    
    say('reading {}'.format(embedding_file))
    
    with open(embedding_file) as f:
        for i, line in enumerate(f):    
            toks = line.split()
            freqs[i] = toks[0]
            words[i] = toks[1]
            rep[i] = map(lambda x: float(x), toks[2:])
    
    say('total {} embeddings of dimension {}'.format(len(rep), len(rep[rep.keys()[0]])))            

    A = numpy.zeros((len(rep), len(rep[rep.keys()[0]])))
    for i in range(len(rep)):
        A[i,:] = rep[i]
  
    return freqs, words, A


def write_embeddings(freqs, words, matrix, filename):

    with open(filename, 'wb') as outf:
        for i in range(len(words)):
            print >> outf, freqs[i], words[i],
            for val in matrix[i,:]:
                print >> outf, val,
            print >> outf
    

def normalize(embedding_file, target):

    assert(target == 'rows' or target == 'columns')
    
    freqs, words, A = read_embeddings(embedding_file)    

    if target == 'columns':
        say('normalizing columns')
        
        for j in range(A.shape[1]):
            A[:,j] /= numpy.linalg.norm(A[:,j])
    
        write_embeddings(freqs, words, A, embedding_file + '.cols_normalized')
    
    elif target == 'rows':
        say('normalizing rows')

        for i in range(A.shape[0]):
            A[i,:] /= numpy.linalg.norm(A[i,:])
            
        write_embeddings(freqs, words, A, embedding_file + '.rows_normalized')
    
    
def command(command_str):
    say(command_str)
    os.system(command_str)
