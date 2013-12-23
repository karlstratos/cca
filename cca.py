import argparse
from src.io import command
from src.io import set_quiet
from src.strop import count_unigrams
from src.strop import decide_vocab
from src.strop import extract_views
from src.canon import canon

def main(args):
    set_quiet(args.quiet)
    
    if args.corpus: 
        assert(args.cutoff is not None)
        unigrams = count_unigrams(args.corpus)
        vocab, outfname = decide_vocab(unigrams, args.cutoff, args.myvocab)
        extract_views(args.corpus, vocab, outfname)
    
    if args.views:
        assert(args.m is not None and args.kappa is not None)
        C = canon()     
        C.get_stats(args.views)        
        C.set_params(args.m, args.kappa)
        C.start_logging()        
        C.approx_cca()
        C.write_result()
        C.end_logging()
    
    if args.clean:
        command('find . -type f -name \'*.pyc\' -print -o -name \'*~\' -print | xargs rm -rf') # remove *.pyc *~ 
        command('find input/example/ -type f -not -name \'example.corpus\' | xargs rm -rf')    # remove input/example/* except for example.corpus
        command('rm -rf output/example*')                                                      # remove output/example*

if __name__=='__main__':    
    argparser = argparse.ArgumentParser('Performs various operations related to CCA')
    argparser.add_argument('--corpus',        type=str,             help='count n-grams from this corpus')
    argparser.add_argument('--cutoff',        type=int,             help='cutoff words appearing <= this number')         
    argparser.add_argument('--myvocab',       type=str,             help='optional: keep words in these tagged sentences even if rare')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--views',         type=str,             help='views to do CCA on')
    argparser.add_argument('--m',             type=int,             help='number of CCA dimensions')
    argparser.add_argument('--kappa',         type=int,             help='smoothing parameter')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--clean',         action='store_true',  help='clean up the project folder and remove sample outputs')
    argparser.add_argument('--quiet',         action='store_true',  help='quiet mode')
    args = argparser.parse_args()
    main(args)
    
