import os
import re
import cPickle
import datetime
from time import strftime
from io import say
from io import wc_l
from io import inline_print
from pca import pca_svd
from svd import mysparsesvd
from numpy import array
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from collections import Counter

class canon(object):

    def get_stats(self, views):        
        def update_mapping(raw_feat_tok, smap, imap, featval, head):
            feat_obj = raw_feat_tok.split('<val>')
            featstr = feat_obj[0]
            if not featstr in smap:
                smap[featstr] = head
                imap[head] = featstr
                featval[head] = 1 if len(feat_obj) == 1 else float(feat_obj[1]) 
                head += 1
            return smap[featstr], head            
        
        assert(os.path.isfile(views))
        say('views: {}'.format(views))
        self.views = views
        
        pickle_file = self.views + '.pickled'
        if os.path.isfile(pickle_file):
            with open(pickle_file) as f:
                self.sX, self.sY, self.iX, self.iY, self.massXY, self.sqmassX, self.sqmassY, self.M = cPickle.load(f)
            return
        
        say('Gathering statistics from: %s' % self.views)
        self.sX = {}
        self.sY = {}
        self.iX = {}
        self.iY = {}
    
        self.massXY = Counter()
        self.sqmassX = Counter()
        self.sqmassY = Counter()
        self.M = 0. # number of samples
        matchObj = re.match( r'(.*)window(\d)(.*)', views)
        window_size = int(matchObj.group(2))
        
        x_head = 0 # distinct number per feature in view 1 
        y_head = 0 # distinct number per feature in view 2
        featvalX = {}
        featvalY = {}
        
        num_lines = wc_l(self.views)
        linenum = 0
        with open(self.views) as f:
            for line in f:
                linenum += 1
                toks = line.split()
                count = int(toks[0])
                self.M += count                
                
                curtain = toks.index('|:|')
                
                # view 1 features
                xs = []
                for i in range(1, curtain):
                    x, x_head = update_mapping(toks[i], self.sX, self.iX, featvalX, x_head)
                    self.sqmassX[x] += count * pow(featvalX[x], 2)
                    xs.append(x)

                # view 2 features
                for i in range(curtain+1, len(toks)):
                    y, y_head = update_mapping(toks[i], self.sY, self.iY, featvalY, y_head)
                    self.sqmassY[y] += count * pow(featvalY[y], 2)
                    for x in xs:
                        self.massXY[x, y] += count * featvalX[x] * featvalY[y]
                
                if linenum % 1000 is 0:
                    inline_print('Processing line %i of %i' % (linenum, num_lines))
        
        inline_print('\n')
        self.massXY = csc_matrix((self.massXY.values(), zip(*self.massXY.keys())), shape=(len(self.sqmassX), len(self.sqmassY)))
        self.sqmassX = array([self.sqmassX[j] for j in range(len(self.sqmassX))]) / (window_size - 1)
        self.sqmassY = array([self.sqmassY[j] for j in range(len(self.sqmassY))])
        self.M /= window_size - 1            

        with open(pickle_file, 'wb') as outf:
            cPickle.dump((self.sX, self.sY, self.iX, self.iY, self.massXY, self.sqmassX, self.sqmassY, self.M), outf, protocol=cPickle.HIGHEST_PROTOCOL) 
    
    def set_params(self, m, kappa):
        say('m: {}'.format(m))
        say('kappa: {}'.format(kappa))
        self.m = m
        self.kappa = kappa

    def rec(self, string, newline=True):
        if newline:
            print >> self.logf, string
            self.logf.flush()
        else:
            print string,
            self.logf.flush()
        say(string)        
                
    def start_logging(self):
        self.dirname = 'output/{}.m{}.kappa{}'.\
                        format(os.path.basename(self.views), self.m, self.kappa)
        self.dirname += '.out'
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)                
        self.logf = open(self.dirname+'/log', 'wb')
        self.start_time = datetime.datetime.now().replace(microsecond=0)
        self.rec(strftime('\nStart time: %Y-%m-%d %H:%M:%S'))
        self.rec('________________________________________')

    def end_logging(self):
        self.rec('________________________________________')
        self.rec(strftime('End time: %Y-%m-%d %H:%M:%S'))
        self.end_time = datetime.datetime.now().replace(microsecond=0)
        self.rec('Total time: ' + str(self.end_time - self.start_time))
        self.logf.close()

    def approx_cca(self):        
        def compute_invsqrt_diag_cov(sqmass, kappa, M):
            smoothed_variances = (sqmass + kappa) / M 
            diags = [i for i in range(len(sqmass))]
            invsqrt_cov = csc_matrix((pow(smoothed_variances, -.5), (diags, diags)), shape=(len(sqmass), len(sqmass)))     
            return invsqrt_cov
                        
        self.rec('\nPerform approximate CCA:\n1. Pseudo-whitening')    
        invsqrt_covX = compute_invsqrt_diag_cov(self.sqmassX, self.kappa, self.M)
        invsqrt_covY = compute_invsqrt_diag_cov(self.sqmassY, self.kappa, self.M)

        C = (1./self.M) * invsqrt_covX * self.massXY * invsqrt_covY # still sparse
        
        self.rec('\tC has dimensions {} x {} ({} nonzeros)'.format(C.shape[0], C.shape[1], C.nnz))

        self.rec('2. Exact thin SVD on C')
        U, self.corr, _ = mysparsesvd(C, self.m)
    
        self.rec('3. De-whitening')
        self.A = invsqrt_covX * U;
    
    def write_result(self):
        self.write_corr()
        self.write_A()

    def write_corr(self):
        say('\nNormalizing correlation coefficients and storing them at: %s' % self.dirname+'/corr')
        self.corr = self.corr / max(self.corr)
        with open(self.dirname+'/corr', 'wb') as f:
            for val in self.corr:
                print >> f, val 
        
    def write_A(self):
        sorted_indices = self.sqmassX.argsort()[::-1]

        def write_embeddings(outfilename, A):
            with open(outfilename, 'wb') as f:
                for i in sorted_indices:
                    print >> f, self.sqmassX[i], self.iX[i], 
                    for j in range(len(A[i,:])):
                        print >> f, A[i,j], 
                    print >> f            

        self.rec('Storing A at: %s' % self.dirname+'/A')
        write_embeddings(self.dirname+'/A', self.A)

        self.rec('Storing A.rows_normalized at: %s' % self.dirname+'/A.rows_normalized')
        Atemp = self.A
        for i in range(Atemp.shape[0]):
            Atemp[i,:] /= norm(Atemp[i,:])
        write_embeddings(self.dirname+'/A.rows_normalized', Atemp)
            
        self.rec('Storing A.rows_normalized.pca100 at: %s' % self.dirname+'/A.rows_normalized.pca100')
        pca_trans, _, _ = pca_svd(Atemp)
        Atemp = pca_trans[:,:100]
        write_embeddings(self.dirname+'/A.rows_normalized.pca100', Atemp)
        


