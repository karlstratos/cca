import os
import cPickle
import datetime
from time import strftime
from io import say
from io import wc_l
from io import inline_print
from io import write_row
from io import complete_path
from svd import mysparsesvd
from numpy import array
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from collections import Counter

_rare_ = '<?>'

class canon(object):

    def set_params(self, m, kappa):        
        say('m: {}'.format(m))
        say('kappa: {}'.format(kappa))
        self.m = m
        self.kappa = kappa

    def get_stat(self, stat):
        self.stat = complete_path(stat)
        XYstats = self.stat + 'XY'
        Xstats = self.stat + 'X' 
        Ystats = self.stat + 'Y' 
        
        assert(os.path.isfile(XYstats) and 
               os.path.isfile(Xstats) and 
               os.path.isfile(Ystats))

        say('XYstats: {}'.format(XYstats))
        say('Xstats: {}'.format(Xstats))
        say('Ystats: {}'.format(Ystats))
        self.wordmap = {}
        wordmapf = self.stat + 'wordmap'
        with open(wordmapf) as f:
            for line in f:
                toks = line.split()
                self.wordmap[int(toks[0])-1] = toks[1]
        
        pickle_file = self.stat + 'pickle'
        if os.path.isfile(pickle_file):
            with open(pickle_file) as f:
                self.countXY, self.countX, self.countY, self.num_samples = \
                    cPickle.load(f)
            return
        
        self.countXY = Counter()
        self.countX = Counter()
        self.countY = Counter()
        self.num_samples = 0. 
        
        num_lines = wc_l(XYstats)
        linenum = 0
        with open(XYstats) as f:
            for line in f:
                linenum += 1
                toks = line.split()
                x, y, count = int(toks[0])-1, int(toks[1])-1, int(toks[2])
                self.countXY[x, y] = count 
                if linenum % 1000 is 0: 
                    inline_print('Processing line %i of %i' % 
                                 (linenum, num_lines))
        
        with open(Xstats) as f:
            for line in f:
                toks = line.split()
                x, count = int(toks[0])-1, int(toks[1])
                self.countX[x] = count
                self.num_samples += count
        
        with open(Ystats) as f:
            for line in f:
                toks = line.split()
                y, count = int(toks[0])-1, int(toks[1])
                self.countY[y] = count
        
        inline_print('\nConstructing matrices\n')
        self.countXY = csc_matrix((self.countXY.values(), 
                                   zip(*self.countXY.keys())), 
                                  shape=(len(self.countX), len(self.countY)))
        self.countX = array([self.countX[i] for i in range(len(self.countX))])
        self.countY = array([self.countY[i] for i in range(len(self.countY))])

        with open(pickle_file, 'wb') as outf:
            cPickle.dump((self.countXY, self.countX, self.countY, 
                          self.num_samples), outf, 
                         protocol=cPickle.HIGHEST_PROTOCOL) 
    
    def rec(self, string, newline=True):
        if newline:
            print >> self.logf, string
            self.logf.flush()
        else:
            print string,
            self.logf.flush()
        say(string)        
                
    def start_logging(self):
        self.dirname = 'output/{}.m{}.kappa{}.out'.\
                        format(self.stat[:-1].rsplit('/',1)[1] , 
                               self.m, self.kappa)
        if not os.path.exists(self.dirname): os.makedirs(self.dirname)                
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
        def compute_invsqrt_diag_cov(count, kappa, num_samples):
            p1 = (count + kappa) / float(num_samples) 
            diags = [i for i in range(len(count))]
            invsqrt_cov = csc_matrix((pow(p1, -.5), (diags, diags)), 
                                     shape=(len(count), len(count)))     
            return invsqrt_cov
                        
        invsqrt_covX = compute_invsqrt_diag_cov(self.countX, 
                                                self.kappa, self.num_samples)
        invsqrt_covY = compute_invsqrt_diag_cov(self.countY, 
                                                self.kappa, self.num_samples)
        Omega = float(1./self.num_samples) * \
            invsqrt_covX * self.countXY * invsqrt_covY # still sparse
        
        self.rec('Omega: dimensions {} x {}, {} nonzeros'
                 .format(Omega.shape[0], Omega.shape[1], Omega.nnz))
        self.rec('Computing {} left singular vectors U of Omega...'
                 .format(self.m))
        self.U, self.sv, _ = mysparsesvd(Omega, self.m)
        for i in range(self.U.shape[0]): self.U[i,:] /= norm(self.U[i,:])
    
    def write_result(self):
        self.write_sv()
        self.write_U()

    def write_sv(self):
        say('\nStoring singular values at: %s' % self.dirname+'/sv')
        with open(self.dirname+'/sv', 'wb') as outf:
            for i in range(len(self.sv)): print >> outf, self.sv[i]
        
    def write_U(self):
        say('Storing row-normalized U at: %s' % self.dirname+'/Ur')
        sorted_indices = [pair[0] for pair in sorted([(i, self.countX[i]) 
                                                      for i in self.wordmap], 
                                                     key=lambda x:x[1], 
                                                     reverse=True)]
        with open(self.dirname+'/Ur', 'wb') as f:
            for i in sorted_indices: write_row(f, self.countX[i], 
                                               self.wordmap[i], self.U[i,:]) 
