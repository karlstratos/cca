import os
import argparse
from src.tools import set_quiet
from src.tools import count_ngrams
from src.tools import cutoff_rare
from src.tools import extract_views
from src.tools import pairwise_classify
from src.tools import perform_pca
from src.canon import canon


description = 'Performs various operations related to CCA'
argparser = argparse.ArgumentParser(description)

argparser.add_argument('--corpus', type=str, help='count n-grams from this corpus')
#_________________________________________________________________________________________________________________________________
argparser.add_argument('--ngrams',        type=str,            help='an n-gram file')
argparser.add_argument('--cutoff',        type=int,            help='cutoff words appearing <= this number')         
argparser.add_argument('--unigrams',      type=str,            help='unigrams needed for identifying rare words')
argparser.add_argument('--myvocab',       type=str,            help='tagged sentences: keep these words even if rare')
argparser.add_argument('--extract_views', action='store_true', help='extract views from n-grams')
argparser.add_argument('--outfile',       type=str,            help='store the preprocessed output in this file')
#_________________________________________________________________________________________________________________________________
argparser.add_argument('--views',     type=str,                           help='views to do CCA on')
argparser.add_argument('--cca_dim',   type=int,            default=200,   help='number of CCA dimensions')
argparser.add_argument('--extra_dim', type=int,            default=40,    help='oversampling parameter')
argparser.add_argument('--power_num', type=int,            default=5,     help='number of power iterations')
argparser.add_argument('--kappa',     type=int,            default=100,   help='pseudocounts for smoothing')
argparser.add_argument('--wantB',     action='store_true',                help='write view 2 projeciton as well?')
argparser.add_argument('--optimize',  action='store_true',                help='optimize params on anchor words classification')
#_________________________________________________________________________________________________________________________________
argparser.add_argument('--A', type=str, help='view 1 embeddings')
argparser.add_argument('--anchor_words', type=str, help='list of anchor words to classify')
argparser.add_argument('--pca', type=int, help='reduce dimension of A down to this number using PCA')
#_________________________________________________________________________________________________________________________________
argparser.add_argument('--clean', action='store_true', help='clean up the project folder')
argparser.add_argument('--quiet', action='store_true', help='quiet mode')

args = argparser.parse_args()

set_quiet(args.quiet)

if args.corpus:
    count_ngrams(args.corpus)

if args.ngrams:
    if args.cutoff:
        cutoff_rare(args.ngrams, args.cutoff, args.unigrams, args.myvocab, args.outfile)
    
    if args.extract_views:
        extract_views(args.ngrams, args.outfile)

if args.views:
    
    if args.optimize:
        assert(os.path.isfile(args.anchor_words))
    
    C = canon()
 
    C.set_views(args.views)
    
    C.set_wantB(args.wantB)
    
    C.get_stats() # read in the data only once
    
    if not args.optimize:
        cca_dims   = [args.cca_dim]
        kappas     = [args.kappa]
        extra_dims = [args.extra_dim]
        power_nums = [args.power_num]
    else:
        cca_dims   = [200, 400, 600]
        kappas     = [50, 100, 150]
        extra_dims = [100, 200]
        power_nums = [5, 10]
        best_acc = 0
        dirs = []
        optimal_dir = False

    for cca_dim in cca_dims:
        for kappa in kappas:
            for extra_dim in extra_dims:
                for power_num in power_nums:    
                    C.set_params(cca_dim, kappa, extra_dim, power_num)
    
                    C.start_logging()
                    
                    C.approx_cca()
                    
                    C.write_result()
                        
                    C.end_logging()
                    
                    if args.optimize:
                        dirs.append(C.dirname)
                        acc = pairwise_classify(C.dirname+'/A', args.anchor_words)
                        if acc > best_acc:
                            best_acc = acc
                            best_cca_dim = cca_dim
                            best_kappa = kappa
                            best_extra_dim = extra_dim
                            best_power_num = power_num
                            optimal_dir = C.dirname
                            print 'current best: {} ({}, {}, {}, {})'.format(best_acc, cca_dim, kappa, extra_dim, power_num)    

    if args.optimize:
        for d in dirs:
            if d != optimal_dir:
                print 'removing the suboptimal:', d
                os.system('rm -fr ' + d)

if args.A:
    if args.anchor_words:
        pairwise_classify(args.A, args.anchor_words)
        
    if args.pca:
        perform_pca(args.A, args.pca) 
        
if args.clean:
    os.system('rm -rf src/*.pyc src/*~ *.pyc *~ input/sample/*grams* input/sample/*featurized*')



    