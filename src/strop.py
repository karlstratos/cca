import os
import shutil
from io import say
from io import inline_print
from collections import deque
from collections import Counter

_rare_ = '<?>'
_buffer_ = '<*>'
token_dict = {}

def count_unigrams(corpus):
    unigrams = os.path.splitext(corpus)[0] + '.1grams'
    if not os.path.isfile(unigrams): count_ngrams(corpus, n_vals=[1])
    else: say('{} exists'.format(unigrams))
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
            if not lines: break
            for line in lines:
                toks = line.split()
                for tok in toks:
                    num_tok += 1
                    if num_tok % 1000 is 0: inline_print('Processed %i tokens' % (num_tok))
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

def decide_vocab(unigrams, cutoff, vocab_size):
    global token_dict
    assert(unigrams and os.path.isfile(unigrams))     
    assert((not (cutoff is None and vocab_size is None)) and (cutoff is None or vocab_size is None))        

    say('Reading unigrams')
    vocab = {}
    num_words = 0 
    total_sum = 0.
    mysum = 0.
    token_dict = {}
    token_head = 1 # starting from 1 for matlab 
    
    with open(unigrams) as f:
        for line in f:
            num_words += 1
            toks = line.split()
            if len(toks) != 2: continue
            word = toks[0]
            count = int(toks[1])
            total_sum += count            

            if ((cutoff is not None) and (count <= cutoff)) or ((vocab_size is not None) and len(vocab) == vocab_size): continue             
            vocab[word] = count
            if not word in token_dict: token_dict[word] = token_head; token_head += 1
            mysum += count  
    
    if cutoff is not None:
        say('Cutoff %i: keep %i out of %i words (%5.2f%% unigram mass)' % (cutoff, len(vocab), num_words, mysum/total_sum*100))
        outfname = os.path.splitext(unigrams)[0] + '.cutoff' + str(cutoff)
         
    if vocab_size is not None: 
        say('Vocab %i: keep %i out of %i words (%5.2f%% unigram mass)' % (vocab_size, len(vocab), num_words, mysum/total_sum*100))
        outfname = os.path.splitext(unigrams)[0] + '.vocab' + str(vocab_size)
        
    if len(vocab) < num_words: token_dict[_rare_] = len(token_dict) + 1
    
    return vocab, outfname

def extract_stats(corpus, vocab, stats, window):
    stats += '.window' + str(window)    
    assert(os.path.isfile(corpus))
    
    cooccur_count = Counter()
    Xcount = Counter()
    Ycount = Counter()
    def inc_stats(q):
        center = (window - 1) / 2 # position of the current token
        if q[center] == _buffer_: return
        token = q[center] if q[center] in vocab else _rare_
        Xcount[token] += 1
        for i in range(window):
            if i != center:
                if q[i] == _buffer_: continue
                friend = q[i] if q[i] in vocab else _rare_
                rel_position = i-center
                position_marker = '<+'+str(rel_position)+'>' if rel_position > 0 else '<'+str(rel_position)+'>'
                friend += position_marker
                cooccur_count[(token, friend)] += 1
                Ycount[friend] += 1
            
    num_tok = 0
    q = deque([_buffer_ for _ in range(window-1)], window)
    with open(corpus) as f:
        while True:
            lines = f.readlines(10000000) # caching lines
            if not lines: break
            for line in lines:
                toks = line.split()
                for tok in toks:
                    num_tok += 1
                    if num_tok % 1000 is 0: inline_print('Processed %i tokens' % (num_tok))
                    q.append(tok)
                    inc_stats(q)                    
                 
    for _ in range(window-1):
        q.append(_buffer_)
        inc_stats(q)

#     for token, friend in cooccur_count: print token, friend, cooccur_count[(token, friend)] 
#     print 
#     for token in Xcount: print token, Xcount[token] 
#     print 
#     for friend in Ycount: print friend, Ycount[friend]
#     exit()

    say('\nCreating directory {}'.format(stats))
    if os.path.exists(stats): shutil.rmtree(stats)
    os.makedirs(stats)
    friend_dict = {}
    friend_head = 1 # starting from 1 for matlab     
    
    with open(stats + '/XY', 'wb') as XYfile:
        for (token, friend) in cooccur_count:
            if not friend in friend_dict: friend_dict[friend] = friend_head; friend_head += 1 
            print >> XYfile, token_dict[token], friend_dict[friend], cooccur_count[(token, friend)]

    with open(stats + '/X', 'wb') as Xfile:
        for token in Xcount: print >> Xfile, token_dict[token], Xcount[token]
 
    with open(stats + '/Y', 'wb') as Yfile:
        for friend in Ycount: print >> Yfile, friend_dict[friend], Ycount[friend]

    with open(stats + '/wordmap', 'wb') as wordmapfile:
        for token in token_dict: print >> wordmapfile, token_dict[token], token
            
        
        