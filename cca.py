import argparse
from src.io import clean
from src.io import set_quiet
from src.strop import count_unigrams
from src.strop import decide_vocab
from src.strop import extract_stats
from src.canon import canon

def main(args):
    set_quiet(args.quiet)
    
    if args.corpus: 
        unigrams = count_unigrams(args.corpus)
        vocab, outfname = decide_vocab(unigrams, args.cutoff, args.vocab)
        extract_stats(args.corpus, vocab, outfname, args.window)
    
    if args.stats:        
        C = canon()
        C.set_params(args.m, args.kappa)     
        C.get_stats(args.stats)        
        C.start_logging()
        C.approx_cca()
        C.end_logging()
        C.write_result()
        
    if args.clean: clean()
    
if __name__=='__main__':    
    argparser = argparse.ArgumentParser('Derives word vectors by decomposing a scaled covariance matrix')
    argparser.add_argument('--corpus',        type=str,             help='count words from this corpus')
    argparser.add_argument('--cutoff',        type=int,             help='cut off words appearing <= this number')
    argparser.add_argument('--vocab',         type=int,             help='size of the vocabulary')
    argparser.add_argument('--window',        type=int, default=3,  help='size of the sliding window')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--stats',         type=str,             help='directory containing statistics')
    argparser.add_argument('--m',             type=int,             help='number of dimensions')
    argparser.add_argument('--kappa',         type=int,             help='smoothing parameter')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--clean',         action='store_true',  help='clean up directories')
    argparser.add_argument('--quiet',         action='store_true',  help='quiet mode')
    args = argparser.parse_args()
    main(args)
    
