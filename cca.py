import argparse
from src.tools import count_ngrams
from src.tools import extract_views
from src.canon import canon

description = 'Performs various operations required for CCA'
argparser = argparse.ArgumentParser(description)

# for making n-grams
argparser.add_argument('--count_ngrams',  type=str,                           help='count n-grams from this corpus')

# for featurizing n-grams
argparser.add_argument('--extract_views', type=str,                           help='extract views from these n-grams')
argparser.add_argument('--threshold',     type=int,            default=0,     help='discard features that appear <= this number')
argparser.add_argument('--spelling',      action='store_true', default=False, help='use spelling features when extracting views')

# for CCA 
argparser.add_argument('--views',     type=str,                           help='views to do CCA on')
argparser.add_argument('--cca_dim',   type=int,            default=200,   help='number of CCA dimensions')
argparser.add_argument('--extra_dim', type=int,            default=40,    help='oversampling parameter')
argparser.add_argument('--power_num', type=int,            default=5,     help='number of power iterations')
argparser.add_argument('--kappa',     type=int,            default=100,   help='pseudocounts for smoothing')
argparser.add_argument('--wantB',     action='store_true', default=False, help='write view 2 projeciton as well?')
args = argparser.parse_args()


if args.count_ngrams:    
    count_ngrams(args.count_ngrams)


if args.extract_views:
    extract_views(args.extract_views, args.threshold, args.spelling)    


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
    
    
    