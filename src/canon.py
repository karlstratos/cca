import os
import re
import cPickle
import datetime
from time import strftime
from io import say
from io import wc_l
from io import inline_print
from io import write_row
from svd import mysparsesvd
from numpy import array
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from collections import Counter

_rare_ = '<?>'

class canon(object):

    def get_stats(self, stats):        
        assert(os.path.isfile(stats))
        say('stats: {}'.format(stats))
        self.views = stats
        self.unigrams = os.path.dirname(stats) + '/' + os.path.basename(stats).split('.')[0]+'.1grams'
        assert(os.path.isfile(self.unigrams))
        
        pickle_file = self.views + '.pickled'
        if os.path.isfile(pickle_file):
            with open(pickle_file) as f:
                self.countXY, self.countX, self.countY, self.num_samples = cPickle.load(f)
            return
        
        say('Gathering statistics from: %s' % self.views)
        self.countXY = Counter()
        self.countX = Counter()
        self.countY = Counter()
        self.num_samples = 0. 
        matchObj = re.match( r'(.*)window(\d)(.*)', stats)
        window_size = int(matchObj.group(2))
        
        num_lines = wc_l(self.views)
        linenum = 0
        with open(self.views) as f:
            for line in f:
                linenum += 1
                toks = line.split()
                x, y, count = int(toks[0])-1, int(toks[1])-1, int(toks[2])
                self.num_samples += count                
                self.countX[x] += count
                self.countY[y] += count
                self.countXY[x, y] += count 
                
                if linenum % 1000 is 0: inline_print('Processing line %i of %i' % (linenum, num_lines))
        
        inline_print('\nConstructing matrices\n')
        self.countXY = csc_matrix((self.countXY.values(), zip(*self.countXY.keys())), shape=(len(self.countX), len(self.countY)))
        self.countX = array([self.countX[i] for i in range(len(self.countX))]) / (window_size - 1)
        self.countY = array([self.countY[i] for i in range(len(self.countY))])
        self.num_samples /= window_size - 1
        
        with open(pickle_file, 'wb') as outf:
            cPickle.dump((self.countXY, self.countX, self.countY, self.num_samples), outf, protocol=cPickle.HIGHEST_PROTOCOL) 
    
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
        def compute_invsqrt_diag_cov(count, kappa, M):
            p1 = (count + kappa) / M 
            diags = [i for i in range(len(count))]
            invsqrt_cov = csc_matrix((pow(p1, -.5), (diags, diags)), shape=(len(count), len(count)))     
            return invsqrt_cov
                        
        invsqrt_covX = compute_invsqrt_diag_cov(self.countX, self.kappa, self.num_samples)
        invsqrt_covY = compute_invsqrt_diag_cov(self.countY, self.kappa, self.num_samples)
        C = (1./self.num_samples) * invsqrt_covX * self.countXY * invsqrt_covY # still sparse
        
        self.rec('C: dimensions {} x {}, {} nonzeros'.format(C.shape[0], C.shape[1], C.nnz))
        self.rec('Computing {} left singular vectors U of C...'.format(self.m))
        self.U, self.sv, _ = mysparsesvd(C, self.m)
    
    def write_result(self):
        self.write_sv()
        self.write_U()

    def write_sv(self):
        say('\nStoring singular values at: %s' % self.dirname+'/sv')
        nsv = self.sv / max(self.sv)
        with open(self.dirname+'/sv', 'wb') as outf:
            for i in range(len(self.sv)):
                print >> outf, self.sv[i], nsv[i]
        
    def write_U(self):
        imap = {} # index -> (word, count)        
        vsize = self.U.shape[0]-1
        raresum = 0 # accumulate counts for the symbol _rare_
        with open(self.unigrams) as uni:
            for i, line in enumerate(uni):
                word, countstr = line.split(); count = int(countstr)
                if i < vsize: imap[i] = (word, count)
                else: raresum += count
        imap[vsize] = (_rare_, raresum)

        say('Storing row-normalized U at: %s' % self.dirname+'/Ur')
        for i in range(vsize): self.U[i,:] /= norm(self.U[i,:])
        with open(self.dirname+'/Ur', 'wb') as f:
            no_rare = True
            for i in range(vsize):
                if no_rare and imap[i][1] <= imap[vsize][1]:
                    write_row(f, imap[vsize][1], imap[vsize][0], self.U[vsize,:])
                    no_rare = False
                write_row(f, imap[i][1], imap[i][0], self.U[i,:]) 
