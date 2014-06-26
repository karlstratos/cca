import os
import sys
from src.call_matlab import call_matlab
from src.canon import canon
from src.io import clean
from src.io import set_quiet
from src.strop import count_unigrams
from src.strop import decide_vocab
from src.strop import extract_stat

global corpus, cutoff, window, gold_Xcount, gold_Ycount, gold_XYcount
set_quiet(True)

# the dog saw the cat
# the dog barked
# the cat meowed
corpus = 'input/example/example.corpus'

# Check the counts in the above global variables.
def check():
    unigrams = count_unigrams(corpus)
    vocab, outfname = decide_vocab(unigrams, cutoff, None, None)
    XYcount, Xcount, Ycount, stat = extract_stat(corpus, vocab, 
                                                 outfname, window)
    for x in Xcount: assert(Xcount[x] == gold_Xcount[x])
    for y in Ycount: assert(Ycount[y] == gold_Ycount[y])
    for x, y in XYcount: assert(XYcount[x,y] == gold_XYcount[x,y])
    return stat 

# Case 1: cutoff = 0, window = 2
cutoff = 0
window = 2

gold_Xcount = {'the': 4, 
               'dog': 2, 
               'cat': 2, 
               'saw': 1, 
               'barked': 1, 
               'meowed': 1
               }

gold_Ycount = {'the<+1>': 3, 
               'dog<+1>': 2, 
               'cat<+1>': 2, 
               'saw<+1>': 1, 
               'barked<+1>': 1, 
               'meowed<+1>': 1
               }

gold_XYcount = {('the','dog<+1>'): 2, 
                ('the','cat<+1>'): 2, 
                ('dog','saw<+1>'): 1, 
                ('dog','barked<+1>'): 1,
                ('cat','the<+1>'): 1, 
                ('cat','meowed<+1>'): 1, 
                ('barked','the<+1>'): 1, 
                ('saw','the<+1>'): 1
                }

check()

# Case 2: cutoff = 0, window = 3
window = 3
gold_Ycount = {'the<+1>': 3, 
               'the<-1>': 4, 
               'dog<+1>': 2, 
               'dog<-1>': 2, 
               'cat<+1>': 2, 
               'cat<-1>': 2,
               'saw<+1>': 1, 
               'saw<-1>': 1, 
               'barked<+1>': 1, 
               'barked<-1>': 1, 
               'meowed<+1>': 1
               }

gold_XYcount = {('dog','the<-1>'): 2, 
                ('cat','the<-1>'): 2,
                ('saw','the<+1>'): 1, 
                ('cat','the<+1>'): 1,
                ('barked','the<+1>'): 1, 
                ('saw','dog<-1>'): 1, 
                ('barked','dog<-1>'): 1, 
                ('the','dog<+1>'): 2,
                ('the','saw<-1>'): 1, 
                ('dog','saw<+1>'): 1, 
                ('the','cat<-1>'): 1, 
                ('meowed','cat<-1>'): 1,
                ('the','cat<+1>'): 2,
                ('the', 'barked<-1>'): 1, 
                ('dog','barked<+1>'): 1,
                ('cat','meowed<+1>'):1 
                }

check()

# Case 3: cutoff = 1, window = 3
cutoff = 1
window = 3
gold_Xcount = {'the': 4, 
               'dog': 2,
               'cat': 2,
               '<?>': 3
               }

gold_Ycount = {'the<-1>': 4, 
               'the<+1>': 3, 
               'dog<-1>': 2, 
               'dog<+1>': 2, 
               'cat<-1>': 2, 
               'cat<+1>': 2, 
               '<?><-1>': 2, 
               '<?><+1>': 3
               }

gold_XYcount = {('dog','the<-1>'): 2, 
                ('cat','the<-1>'): 2, 
                ('<?>','the<+1>'): 2, 
                ('cat','the<+1>'): 1,
                ('<?>','dog<-1>'): 2, 
                ('the','dog<+1>'): 2, 
                ('the','<?><-1>'): 2, 
                ('dog','<?><+1>'): 2, 
                ('the','cat<-1>'): 1, 
                ('<?>','cat<-1>'): 1, 
                ('the','cat<+1>'): 2, 
                ('cat','<?><+1>'): 1
                }

stat = check()

# Check if the result of python sparsesvd agrees with the result of Matlab.
m = 2
kappa = 1

C = canon()
C.set_params(m, kappa)
C.get_stat(stat)        
C.start_logging()
C.approx_cca()
C.end_logging()
C.write_result()

outdirname = call_matlab(stat, m, kappa)
sv_matlab = map(lambda line: float(line.split()[0]), 
                open(os.path.join(outdirname, 'sv')).readlines())
for i in range(len(C.sv)): assert(abs(C.sv[i] - sv_matlab[i]) < 1e-10) 

sys.stderr.write('Correctness of statistics and svd calculations verified.\n')
sys.stderr.write('Cleaning.\n')
clean()
