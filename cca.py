import argparse
from src.tools import count_ngrams
from src.tools import cutoff_rare
from src.tools import extract_views
from src.tools import nv_classify
from src.canon import canon

description = 'Performs various operations required for CCA'
argparser = argparse.ArgumentParser(description)

# for counting n-grams from a corpus
argparser.add_argument('--corpus', type=str, help='count n-grams from this corpus')

# n-grams to use 
argparser.add_argument('--ngrams',      type=str,             help='an n-gram file')

# cutoff rare words in the given n-grams
argparser.add_argument('--cutoff_rare', action='store_true',  help='replace rare words with \"<?>\" in these n-grams')
argparser.add_argument('--cutoff',      type=int,  default=0, help='discard features that appear <= cutoff')
argparser.add_argument('--unigrams',    type=str,             help='unigrams needed for identifying rare words')

# for featurizing n-grams
argparser.add_argument('--extract_views', action='store_true', help='extract views from these n-grams')

# for performing CCA 
argparser.add_argument('--views',     type=str,                           help='views to do CCA on')
argparser.add_argument('--cca_dim',   type=int,            default=200,   help='number of CCA dimensions')
argparser.add_argument('--extra_dim', type=int,            default=40,    help='oversampling parameter')
argparser.add_argument('--power_num', type=int,            default=5,     help='number of power iterations')
argparser.add_argument('--kappa',     type=int,            default=100,   help='pseudocounts for smoothing')
argparser.add_argument('--wantB',     action='store_true', default=False, help='write view 2 projeciton as well?')

# for checking the quality of CCA
argparser.add_argument('--A', type=str, help='view 1 embeddings')
argparser.add_argument('--nv_list', type=str, help='list of nouns and verbs to classify')

args = argparser.parse_args()


if args.corpus:
    count_ngrams(args.corpus)


if args.ngrams:
    if args.cutoff_rare:
        cutoff_rare(args.ngrams, args.cutoff, args.unigrams)
    
    if args.extract_views:
        extract_views(args.ngrams)


if args.views:
    C = canon()
        
    C.set_views(args.views)
    
    C.set_wantB(args.wantB)
    
    C.set_params(args.cca_dim, args.kappa, args.extra_dim, args.power_num)
    
    C.start_logging()
    
    C.get_stats()
    
    C.approx_cca()
    
    C.write_result()
        
    C.end_logging()


if args.A:
    if args.nv_list:
        nv_classify(args.A, args.nv_list)
    
    