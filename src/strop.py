import os
from io import say
from io import wc_l
from io import inline_print
from io import scrape_words
from collections import deque
from collections import Counter

_buffer_ = '<*>'
_rare_ = '<?>'

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
    files = [os.path.dirname(corpus)+'/'+os.path.splitext(os.path.basename(corpus))[0]+'.'+str(n)+'grams' for n in n_vals]        
    for i in range(len(n_vals)):
        say('Sorting {} {}grams and writing to: {}'.format(len(ngrams[i]), n_vals[i], files[i]))
        sorted_ngrams = sorted(ngrams[i].items(), key=lambda x: x[1], reverse=True)
        with open(files[i], 'wb') as outf:
            for ngram, count in sorted_ngrams:
                for tok in ngram:
                    print >> outf, tok,
                print >> outf, count

def cutoff_rare(ngrams, cutoff, unigrams, given_myvocab):
    assert(unigrams and os.path.isfile(unigrams)) 
    outfname = ngrams + '.cutoff' + str(cutoff) 
    
    if(given_myvocab):
        myvocab = scrape_words(given_myvocab)
        myvocab_hit = {}
        outfname += '.' + os.path.splitext(os.path.basename(given_myvocab))[0]
    
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
    num_lines = wc_l(ngrams)
    linenum = 0
    with open(ngrams) as f:
        for line in f:
            linenum += 1
            toks = line.split()
            ngram = toks[:-1]
            n = len(ngram)
            count = int(toks[-1])
            new_ngram = []
            for gram in ngram:
                this_tok = gram if gram in vocab else _rare_
                new_ngram.append(this_tok)
            new_ngrams[tuple(new_ngram)] += count
            if linenum % 1000 is 0:
                inline_print('Processing line %i of %i' % (linenum, num_lines))
        
    say('\nSorting {} {}grams and writing to: {}'.format(len(new_ngrams), n, outfname))
    sorted_ngrams = sorted(new_ngrams.items(), key=lambda x: x[1], reverse=True)
    with open(outfname, 'wb') as outf:
        for ngram, count in sorted_ngrams:
            for gram in ngram:
                print >> outf, gram,
            print >> outf, count

def extract_views(ngrams):
    outfname = ngrams + '.featurized'
    say('Writing the featurized file to: ' + outfname)
    
    num_lines = wc_l(ngrams)
    linenum = 0    
    with open(outfname, 'wb') as outf:
        with open(ngrams) as f:
            for line in f:
                linenum += 1
                toks = line.split()
                ngram = toks[:-1]
                count = int(toks[-1])
                center = len(ngram) / 2 # position of the current word
                print >> outf, count,

                view1_holder = phi(ngram[center], 0)
                for view1f in view1_holder:
                    print >> outf, view1f,
                    
                print >> outf, '|:|',
                
                for i in range(len(ngram)): 
                    if i != center:
                        view2_holder = phi(ngram[i], i-center)
                        for view2f in view2_holder:
                            print >> outf, view2f,
                print >> outf
                if linenum % 1000 is 0:
                    inline_print('Processing line %i of %i' % (linenum, num_lines))
    inline_print('\n')

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

def spelling_phi(token):
    holder = {}
    holder['p1='+token[0]] = True
    holder['s1='+token[-1]] = True
    if len(token) > 1:
        holder['p2='+token[:2]] = True
        holder['s2='+token[-2:]] = True
    if len(token) > 2:
        holder['p3='+token[:3]] = True
        holder['s3='+token[-3:]] = True
    if len(token) > 3:
        holder['p4='+token[:4]] = True
        holder['s4='+token[-4:]] = True
    return holder
