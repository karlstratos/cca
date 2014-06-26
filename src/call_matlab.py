import os
from io import complete_path
from io import write_row
from io import read_wordmap
from io import read_freqmap
from io import say

# change it as needed to point to the matlab executable on your machine 
matlab = '/Applications/MATLAB_R2013b.app/bin/matlab' 

def call_matlab(stat, m, kappa):
    assert(m is not None and kappa is not None)
        
    outdirname = 'output/{}.m{}.kappa{}.matlab.out'.format(complete_path(stat)[:-1].rsplit('/',1)[1] , m, kappa)
    if not os.path.exists(outdirname): os.makedirs(outdirname)                

    commandstr = matlab + ' -nojvm -nodisplay -nosplash -r ' + '\"approx_cca(\'' + stat + '\',' + str(m) + ',' + str(kappa) + ',\'' + outdirname + '\')\"'
    os.system(commandstr)
    
    say('Postprocessing to sort rows by frequency...') 
    wordmap = read_wordmap(os.path.join(stat, 'wordmap'))
    freqmap = read_freqmap(os.path.join(stat, 'X'))
    sorted_indices = [pair[0] for pair in sorted([(i, freqmap[i]) for i in wordmap], key=lambda x:x[1], reverse=True)]
    
    lines = open(os.path.join(outdirname, 'Ur')).readlines()
    with open(os.path.join(outdirname, 'Ur'), 'wb') as outf:
        for i in sorted_indices: write_row(outf, freqmap[i], wordmap[i], lines[i].split())
    
    return outdirname
