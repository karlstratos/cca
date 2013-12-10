import argparse
from src import tools
from src.canon import canon


def main(args):
    tools.set_quiet(args.quiet)
    
    if args.corpus:
        tools.count_ngrams(args.corpus)
    
    if args.ngrams:
        if args.cutoff:
            tools.cutoff_rare(args.ngrams, args.cutoff, args.unigrams, args.myvocab)        
        if args.extract_views:
            tools.extract_views(args.ngrams)
    
    if args.views:        
        C = canon()     
        C.set_views(args.views)
        C.set_wantB(args.wantB)
        C.get_stats()        
        C.set_params(args.cca_dim, args.kappa, args.extra_dim, args.power_num, args.no_centering)        
        C.start_logging()        
        C.approx_cca()
        C.write_result()
        C.end_logging()
    
    if args.embedding_file:    
        if args.normalize_columns:
            tools.normalize(args.embedding_file, 'columns')        
        if args.normalize_rows:
            tools.normalize(args.embedding_file, 'rows')        
        if args.pca:
            tools.perform_pca(args.embedding_file, args.pca) 
            
    if args.clean:
        tools.command('find . -type f -name \'*.pyc\' -print -o -name \'*~\' -print | xargs rm -rf') # remove *.pyc *~ 
        tools.command('find input/example/ -type f -not -name \'example.corpus\' | xargs rm -rf')    # remove input/example/* except for example.corpus
        tools.command('rm -rf output/example*')                                                      # remove output/example*



if __name__=='__main__':    
    argparser = argparse.ArgumentParser('Performs various operations related to CCA')
    
    argparser.add_argument('--corpus', type=str, help='count n-grams from this corpus')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--ngrams',        type=str,            help='an n-gram file')
    argparser.add_argument('--cutoff',        type=int,            help='cutoff words appearing <= this number')         
    argparser.add_argument('--unigrams',      type=str,            help='unigrams needed for identifying rare words')
    argparser.add_argument('--myvocab',       type=str,            help='optional: keep words in these tagged sentences even if rare')
    argparser.add_argument('--extract_views', action='store_true', help='extract views from n-grams')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--views',        type=str,                           help='views to do CCA on')
    argparser.add_argument('--cca_dim',      type=int,            default=200,   help='number of CCA dimensions')
    argparser.add_argument('--extra_dim',    type=int,            default=40,    help='oversampling parameter')
    argparser.add_argument('--power_num',    type=int,            default=5,     help='number of power iterations')
    argparser.add_argument('--kappa',        type=int,            default=100,   help='pseudocounts for smoothing')
    argparser.add_argument('--wantB',        action='store_true',                help='write view 2 projeciton as well?')
    argparser.add_argument('--no_centering', action='store_true',                help='skip subtracting the mean')
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--embedding_file', type=str, help='file containing embeddings')
    argparser.add_argument('--normalize_columns', action='store_true', help='normalize the columns of the embedding matrix')
    argparser.add_argument('--normalize_rows', action='store_true', help='normalize the rows of the embedding matrix')
    argparser.add_argument('--pca', type=int, help='reduce dimension of embeddings down to this number using PCA')    
    #_________________________________________________________________________________________________________________________________
    argparser.add_argument('--clean', action='store_true', help='clean up the project folder and remove sample outputs')
    argparser.add_argument('--quiet', action='store_true', help='quiet mode')
    
    main(argparser.parse_args())
    
