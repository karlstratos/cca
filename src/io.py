import os
import sys
import subprocess
from numpy import array
from numpy import zeros
from numpy.linalg import norm

_quiet_ = False

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

def command(command_str):
    say(command_str)
    os.system(command_str)
    
def clean():
    command('find . -type f -name \'*.pyc\' -print -o -name \'*~\' -print | xargs rm -rf')             # remove *.pyc *~
    command('find input/example -not -path "input/example" | grep -v "example.corpus" | xargs rm -rf') # remove input/example/* except for example.corpus
    command('rm -rf output/example*')                                                                  # remove output/example*    

def inline_print(string):
    if not _quiet_:
        sys.stderr.write('\r\t%s' % (string))
        sys.stderr.flush()

def wc_l(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def complete_path(path): return path if path[-1] == '/' else path+'/'

def read_embeddings(embedding_file, top=None, vocab=None):
    freqs = {}
    words = {}
    w2i = {}
    i2w = {}
    rep = {}
    
    say('reading {}'.format(embedding_file))
    i = 0
    with open(embedding_file) as f:
        for line in f:    
            toks = line.split()
            if vocab and (not toks[1] in vocab) and (not toks[1] != '<?>'): continue
            freqs[i] = toks[0]
            words[i] = toks[1]
            w2i[toks[1]] = i
            i2w[i] = toks[1]
            end_ind = len(toks) if not top else top + 2
            rep[toks[1]] = array(map(lambda x: float(x), toks[2:end_ind]))
            i += 1
    
    say('total {} embeddings of dimension {}'.format(len(rep), len(rep[rep.keys()[0]])))            
    A = zeros((len(rep), len(rep[rep.keys()[0]])))
    for i in range(len(rep)): A[i,:] = rep[words[i]]
    return freqs, words, w2i, i2w, rep, A 

def write_embeddings(freqs, words, matrix, filename):
    with open(filename, 'wb') as outf:
        for i in range(len(words)):
            write_row(outf, freqs[i], words[i], matrix[i,:])

def write_row(outf, count, word, vector):
    print >> outf, count, word, 
    for val in vector: print >> outf, val,
    print >> outf

def normalize_rows(embedding_file):
    freqs, words, A, _, _ = read_embeddings(embedding_file)    
    say('normalizing rows')
    for i in range(A.shape[0]): A[i,:] /= norm(A[i,:])
    write_embeddings(freqs, words, A, embedding_file + '.rows_normalized')
    
def read_wordmap(wordmap_file):
    wordmap = {}
    lines = open(wordmap_file).readlines()
    for line in lines: 
        toks = line.split() 
        wordmap[int(toks[0])-1] = toks[1]
    return wordmap

def read_freqmap(freqmap_file):
    freqmap = {}
    lines = open(freqmap_file).readlines()
    for line in lines:
        toks = line.split()
        freqmap[int(toks[0])-1] = int(toks[1])
    return freqmap    
    
    
