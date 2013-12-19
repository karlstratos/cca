import argparse
from src.tools import set_quiet
from src.tools import count_ngrams
from src.tools import cutoff_rare
from src.tools import extract_views
from src.tools import normalize_rows
from src.tools import perform_pca
from src.tools import command
from src.canon import canon

def main(args):
    set_quiet(args.quiet)
    
    if args.corpus:
        count_ngrams(args.corpus)
    
    if args.ngrams:
        if args.cutoff:
            cutoff_rare(args.ngrams, args.cutoff, args.unigrams, args.myvocab)        
        if args.extract_views:
            extract_views(args.ngrams)
    
    if args.views:
        assert(args.m and args.kappa)
        C = canon()     
        C.get_stats(args.views)        
        C.set_params(args.m, args.kappa)
        C.start_logging()        
        C.approx_cca()
        C.write_result()
        C.end_logging()
    
    if args.embedding_file:    
        if args.normalize_rows:
            normalize_rows(args.embedding_file, 'rows')        
        if args.pca:
            perform_pca(args.embedding_file, args.pca, args.top) 
            
    if args.clean:
        command('find . -type f -name \'*.pyc\' -print -o -name \'*~\' -print | xargs rm -rf') # remove *.pyc *~ 
        command('find input/example/ -type f -not -name \'example.corpus\' | xargs rm -rf')    # remove input/example/* except for example.corpus
        command('rm -rf output/example*')                                                      # remove output/example*


if __name__=='__main__':    
    argparser = argparse.ArgumentParser('Performs various operations related to CCA')
    argparser.add_argument('--corpus',        type=str,             help='count n-grams from this corpus')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--ngrams',        type=str,             help='an n-gram file')
    argparser.add_argument('--cutoff',        type=int,             help='cutoff words appearing <= this number')         
    argparser.add_argument('--unigrams',      type=str,             help='unigrams needed for identifying rare words')
    argparser.add_argument('--myvocab',       type=str,             help='optional: keep words in these tagged sentences even if rare')
    argparser.add_argument('--extract_views', action='store_true',  help='extract views from n-grams')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--views',         type=str,             help='views to do CCA on')
    argparser.add_argument('--m',             type=int,             help='number of CCA dimensions')
    argparser.add_argument('--kappa',         type=int,             help='smoothing parameter')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--embedding_file', type=str,            help='file containing embeddings')
    argparser.add_argument('--normalize_rows', action='store_true', help='normalize the rows of the embedding matrix')
    argparser.add_argument('--pca',            type=int,            help='reduce dimension of embeddings down to this number using PCA')    
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--clean',          action='store_true', help='clean up the project folder and remove sample outputs')
    argparser.add_argument('--quiet',          action='store_true', help='quiet mode')
    
    main(argparser.parse_args())
    
