import os
from src.io import set_quiet
from src.strop import count_unigrams
from src.strop import decide_vocab
from src.strop import extract_stats
from src.io import clean
from src.canon import canon
from src.call_matlab import call_matlab

global corpus, cutoff, window, gold_Xcount, gold_Ycount, gold_XYcount
set_quiet(True)
corpus = 'input/example/example.corpus'

def check():
    unigrams = count_unigrams(corpus)
    vocab, outfname = decide_vocab(unigrams, cutoff, None)
    XYcount, Xcount, Ycount, stat = extract_stats(corpus, vocab, outfname, window)
    for x in Xcount: assert(Xcount[x] == gold_Xcount[x])
    for y in Ycount: assert(Ycount[y] == gold_Ycount[y])
    for x, y in XYcount: assert(XYcount[x,y] == gold_XYcount[x,y])
    return stat 

cutoff = 0
window = 2
gold_Xcount = {'the':4, 'dog':2, 'cat':2, 'saw':1, 'barked':1, 'meowed':1}
gold_Ycount = {'the<+1>':3, 'dog<+1>':2, 'cat<+1>':2, 'saw<+1>':1, 'barked<+1>':1, 'meowed<+1>':1}
gold_XYcount = {('the','dog<+1>'):2, ('the','cat<+1>'):2, ('dog','saw<+1>'):1, ('dog','barked<+1>'):1, 
                ('cat','the<+1>'):1, ('cat','meowed<+1>'):1, ('barked','the<+1>'):1, ('saw','the<+1>'):1}
check()

window = 3
gold_Ycount = {'the<+1>':3, 'the<-1>':4, 'dog<+1>':2, 'dog<-1>':2, 'cat<+1>':2, 'cat<-1>':2,
               'saw<+1>':1, 'saw<-1>':1, 'barked<+1>':1, 'barked<-1>':1, 'meowed<+1>':1}
gold_XYcount = {('dog','the<-1>'):2, ('cat','the<-1>'):2, ('saw','the<+1>'):1, ('cat','the<+1>'):1,
                ('barked','the<+1>'):1, ('saw','dog<-1>'):1, ('barked','dog<-1>'):1, ('the','dog<+1>'):2,
                ('the','saw<-1>'):1, ('dog','saw<+1>'):1, ('the','cat<-1>'):1, ('meowed','cat<-1>'):1,
                ('the','cat<+1>'):2, ('the', 'barked<-1>'):1, ('dog','barked<+1>'):1, ('cat','meowed<+1>'):1}
check()

cutoff = 1
window = 3
gold_Xcount = {'the':4, 'dog':2, 'cat':2,  '<?>':3}
gold_Ycount = {'the<-1>':4, 'the<+1>':3, 'dog<-1>':2, 'dog<+1>':2, 'cat<-1>':2, 'cat<+1>':2, '<?><-1>':2, '<?><+1>':3}
gold_XYcount = {('dog','the<-1>'):2, ('cat','the<-1>'):2, ('<?>','the<+1>'):2, ('cat','the<+1>'):1,
                ('<?>','dog<-1>'):2, ('the','dog<+1>'):2, ('the','<?><-1>'):2, ('dog','<?><+1>'):2, 
                ('the','cat<-1>'):1, ('<?>','cat<-1>'):1, ('the','cat<+1>'):2, ('cat','<?><+1>'):1}
stat = check()

m = 2
kappa = 1

C = canon()
C.set_params(m, kappa)
C.get_stats(stat)        
C.start_logging()
C.approx_cca()
C.end_logging()
C.write_result()

outdirname = call_matlab(stat, m, kappa)
sv_matlab = map(lambda line: float(line.split()[0]), open(os.path.join(outdirname, 'sv')).readlines())
for i in range(len(C.sv)): assert(abs(C.sv[i] - sv_matlab[i]) < 1e-10) 

clean()