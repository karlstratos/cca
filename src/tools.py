import sys
import os
#import cPickle
from collections import Counter
from collections import deque

_buffer_ = '<*>'
_rare_ = '<?>'
_status_unit_ = 1000000


def phi(token, rel_position, spelling, rare=False):
    position_marker = '' if rel_position == 0 else '<'+str(rel_position)+'>'
    feat = token+position_marker
    if rare and feat in rare:
        feat = _rare_+position_marker
    holder = {feat : True}
    
    if spelling and token != _buffer_:        
        for i in range(4):
            pref = token[:i+1] + '***' 
            
            suff = '***' + token[-(i+1):]
            
            feat = 'pref'+str(i+1)+'='+pref[:4]+position_marker 
            if rare and feat in rare: 
                feat = 'pref'+str(i+1)+'='+_rare_+position_marker
            holder[feat] = True
            
            feat = 'suff'+str(i+1)+'='+suff[-4:]+position_marker 
            if rare and feat in rare: 
                feat = 'suff'+str(i+1)+'='+_rare_+position_marker
            holder[feat] = True

    return holder


def extract_views(ngrams, threshold, spelling):
    say('\nExtracting two views: %s' % ngrams)
    counts1 = Counter()
    counts2 = Counter()
    
    say('\nFirst, counting features to threshold those that appear <= %d' % threshold)
    say('\tUsing spelling features? ' + str(spelling))
    with open(ngrams) as f:
        for line in f:
            toks = line.split()
            ngram = toks[:-1]
            count = int(toks[-1])
            center = len(ngram) / 2 # position of the current word
            
            # definition of view 1
            view1_holder = phi(ngram[center], 0, spelling)
            for view1f in view1_holder:
                counts1[view1f] += count
            
            # definition of view 2
            for i in range(len(ngram)):
                if i != center:
                    view2_holder = phi(ngram[i], i-center, spelling)
                    for view2f in view2_holder:
                        counts2[view2f] += count
    
    say('\nOriginally, have')
    say('\tview 1: {} features'.format(len(counts1)))
    say('\tview 2: {} features'.format(len(counts2)))
    
    rare1 = {}
    for view1f in counts1:
        if counts1[view1f] <= threshold:
            rare1[view1f] = True

    rare2 = {}
    for view2f in counts2:
        if counts2[view2f] <= threshold:
            rare2[view2f] = True    
    
    new_counts1 = Counter()
    new_counts2 = Counter()
    
    spelling_marker = '' if not spelling else '.spelling'
    outfname = ngrams + '.threshold' + str(threshold) + spelling_marker + '.featurized'
    
    say('\nWriting the feature file to: ' + outfname)
    with open(outfname, 'wb') as outf:
        with open(ngrams) as f:
            for line in f:
                toks = line.split()
                ngram = toks[:-1]
                count = int(toks[-1])
                center = len(ngram) / 2 # position of the current word
                print >> outf, count,

                # definition of view 1
                view1_holder = phi(ngram[center], 0, spelling, rare1)
                for view1f in view1_holder:
                    new_counts1[view1f] += count
                    print >> outf, view1f,
                
                print >> outf, '|:|',
                
                # definition of view 2
                for i in range(len(ngram)):
                    if i != center:
                        view2_holder = phi(ngram[i], i-center, spelling, rare2)
                        for view2f in view2_holder:
                            new_counts2[view2f] += count
                            print >> outf, view2f,
                print >> outf

    say('\nAfter thresholding, have')
    say('\tview 1: {} features'.format(len(new_counts1)))
    say('\tview 2: {} features'.format(len(new_counts2)))
                         
                            
def count_ngrams(corpus, n_vals=False):
    assert(os.path.isfile(corpus))
    
    say('\nCounting n-grams from: %s' % corpus)
    
    if n_vals == False:
        answer = raw_input('\nType in the values of n (e.g., \"1 3 5\"): ')        
        n_vals = [int(n) for n in answer.split()]
    
    files = [os.path.dirname(corpus)+'/'+os.path.splitext(os.path.basename(corpus))[0]+'.'+str(n)+'grams' for n in n_vals]
    
    say('\nWill create the following n-gram files:')    
    for f in files:
        say('\t%s' % f)
    
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

    say('\n\nWriting the n-grams')
    for i in range(len(n_vals)):
        sorted_ngrams = sorted(ngrams[i].items(), key=lambda x: x[1], reverse=True)
        with open(files[i], 'wb') as outf:
            #cPickle.dump(sorted_ngrams, outf, protocol=cPickle.HIGHEST_PROTOCOL)
            for ngram, count in sorted_ngrams:
                for tok in ngram:
                    print >> outf, tok,
                print >> outf, count


def say(string, newline=True):
    if newline:
        print string
        sys.stdout.flush()
    else:
        print string,
        sys.stdout.flush()

