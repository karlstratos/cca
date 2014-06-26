import argparse
from src.io import clean
from src.io import set_quiet
from src.strop import count_unigrams
from src.strop import decide_vocab
from src.strop import extract_stat
from src.strop import rewrite_corpus
from src.canon import canon
from src.call_matlab import call_matlab 

def main(args):
    set_quiet(args.quiet)
    
    if args.corpus: 
        unigrams = count_unigrams(args.corpus)
        vocab, outfname = decide_vocab(unigrams, args.cutoff, 
                                       args.vocab, args.want)
        if args.rewrite: 
            rewrite_corpus(args.corpus, vocab, outfname)
        else:
            extract_stat(args.corpus, vocab, outfname, args.window)
    
    if args.stat:
        assert(args.m is not None and args.kappa is not None)
        if args.no_matlab:        
            C = canon()
            C.set_params(args.m, args.kappa)     
            C.get_stat(args.stat)
            C.start_logging()
            C.approx_cca()
            C.end_logging()
            C.write_result()
        else:
            call_matlab(args.stat, args.m, args.kappa)
        
    if args.clean: clean()
    
if __name__=='__main__':    
    argparser = argparse.ArgumentParser('Derives word vectors')
    argparser.add_argument('--corpus', 
                           type=str, 
                           help='count words from this corpus')
    argparser.add_argument('--cutoff',
                           type=int,
                           help='cut off words appearing <= this number')
    argparser.add_argument('--vocab', 
                           type=int, 
                           help='size of the vocabulary')    
    argparser.add_argument('--window', 
                           type=int, 
                           default=3,
                           help='size of the sliding window')
    argparser.add_argument('--want',
                           type=str, 
                           help='want words in this file')
    argparser.add_argument('--rewrite', 
                           action='store_true', 
                           help='rewrite the (processed) corpus and quit')
    argparser.add_argument('--stat', 
                           type=str, 
                           help='directory containing statistics')
    argparser.add_argument('--m', 
                           type=int, 
                           help='number of dimensions')
    argparser.add_argument('--kappa', 
                           type=int,  
                           help='smoothing parameter')
    argparser.add_argument('--clean', 
                           action='store_true', 
                           help='clean up directories')
    argparser.add_argument('--quiet', 
                           action='store_true',
                           help='quiet mode')
    argparser.add_argument('--no_matlab', 
                           action='store_true', 
                           help='do not call matlab - use python sparsesvd')
    args = argparser.parse_args()
    main(args)
    
