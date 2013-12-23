import os
from io import say
from io import inline_print
from io import scrape_words
from collections import deque
from collections import Counter

_buffer_ = '<*>'
_rare_ = '<?>'
_gstart_ = '_START_'
_gend_ = '_END_'

def count_unigrams(corpus):
    unigrams = os.path.splitext(corpus)[0] + '.1grams'
    if not os.path.isfile(unigrams):
        say('creating {}'.format(unigrams))
        count_ngrams(corpus, n_vals=[1])
    else: 
        say('{} exists'.format(unigrams))
    return unigrams 

def count_ngrams(corpus, n_vals=False):
    assert(os.path.isfile(corpus))    
    if n_vals == False:
        answer = raw_input('Type in the values of n (e.g., \"1 3\"): ')        
        n_vals = [int(n) for n in answer.split()]
    
    num_tok = 0
    ngrams = [Counter() for n in n_vals]                                      
    queues = [deque([_buffer_ for _ in range(n-1)], n) for n in n_vals]
    with open(corpus) as f:
        while True:
            lines = f.readlines(10000000) # caching lines
            if not lines:
                break
            for line in lines:
                toks = line.split()
                for tok in toks:
                    num_tok += 1
                    if num_tok % 1000 is 0:
                        inline_print('Processed %i tokens' % (num_tok))
                    for i in range(len(n_vals)):
                        queues[i].append(tok)
                        ngrams[i][tuple(queues[i])] += 1
                 
    for i in range(len(n_vals)):
        for _ in range(n_vals[i]-1):
            queues[i].append(_buffer_)
            ngrams[i][tuple(queues[i])] += 1

    say('\nTotal {} tokens'.format(num_tok))
    files = [os.path.splitext(corpus)[0]+'.'+str(n)+'grams' for n in n_vals]        
    for i in range(len(n_vals)):
        say('Sorting {} {}grams and writing to: {}'.format(len(ngrams[i]), n_vals[i], files[i]))
        sorted_ngrams = sorted(ngrams[i].items(), key=lambda x: x[1], reverse=True)
        with open(files[i], 'wb') as outf:
            for ngram, count in sorted_ngrams:
                for tok in ngram:
                    print >> outf, tok,
                print >> outf, count

def decide_vocab(unigrams, cutoff, given_myvocab):
    outfname = os.path.splitext(unigrams)[0] + '.cutoff' + str(cutoff) 
    assert(unigrams and os.path.isfile(unigrams))     
    if given_myvocab: 
        myvocab = scrape_words(given_myvocab)
        myvocab_hit = {}
        outfname += '.' + os.path.splitext(os.path.basename(given_myvocab))[0]
    else:
        ans = raw_input('Warning: are you sure you don\'t want a vocab? [y] ')
        if ans is not 'y': exit(0) 
        
    say('Reading unigrams')
    vocab = {}
    num_unigrams = 0 
    total_count = 0.
    mine_count = 0.
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
                
            total_count += count
            if count > cutoff or (given_myvocab and word in myvocab): mine_count += count
                
    say('Cutoff %i: will keep %i out of %i words (%5.2f%% of unigram mass)' % (cutoff, len(vocab), num_unigrams, mine_count / total_count * 100))
    if given_myvocab:
        say('\t- They include {} out of {} in my vocab'.format(len(myvocab_hit), len(myvocab)))

    vocab[_buffer_] = True
    vocab[_gstart_] = True # for google n-grams 
    vocab[_gend_] = True    
    
    ans = raw_input('Do you want to proceed with the setting? [y] ')
    if ans is not 'y': exit(0)
    return vocab, outfname

def extract_views(corpus, vocab, views, window_size=3):
    views += '.window' + str(window_size)
    assert(os.path.isfile(corpus))    
    
    cooccur = Counter()
    def inc_cooccur(q):
        center = window_size / 2 # position of the current word
        token = q[center] if q[center] in vocab else _rare_
        if token == _buffer_ or token == _gstart_ or token == _gend_: return
        view1_holder = phi(token, 0)
        for i in range(window_size):
            if i != center:
                token = q[i] if q[i] in vocab else _rare_
                view2_holder = phi(token, i-center)
                for view2f in view2_holder:
                    for view1f in view1_holder:
                        cooccur[(view1f, view2f)] += 1
            
    num_tok = 0
    q = deque([_buffer_ for _ in range(window_size-1)], window_size)
    with open(corpus) as f:
        while True:
            lines = f.readlines(10000000) # caching lines
            if not lines:
                break
            for line in lines:
                toks = line.split()
                for tok in toks:
                    num_tok += 1
                    if num_tok % 1000 is 0:
                        inline_print('Processed %i tokens' % (num_tok))
                    q.append(tok)
                    inc_cooccur(q)                    
                 
    for _ in range(window_size-1):
        q.append(_buffer_)
        inc_cooccur(q)

    say('\nWriting to {}'.format(views))
    with open(views, 'wb') as outf:
        for (v1f, v2f) in cooccur:
            print >> outf, cooccur[(v1f, v2f)], v1f, '|:|', v2f

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

