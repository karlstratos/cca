import numpy
from scipy.linalg import svd

def pca_eig(A):
    M = center_cols(A)
    pca_directions, pca_variances = sorted_eig(numpy.cov(M.T))
    pca_trans = numpy.dot(M, pca_directions)
    return pca_trans, pca_directions, pca_variances

def pca_svd(A):
    M = center_cols(A)
    U, S, VT = svd(M, full_matrices=False)
    pca_trans = numpy.dot(U, numpy.diag(S))
    pca_directions = VT.T
    pca_variances = pow(S,2)/(A.shape[0]-1)     
    return pca_trans, pca_directions, pca_variances

def sorted_eig(matrix):
    [evals,evecs] = numpy.linalg.eig(matrix)
    idx = evals.argsort()[::-1]   
    evecs = evecs[:,idx] 
    evals = evals[idx]
    return evecs, evals

def center_cols(matrix):
    return matrix - numpy.mean(matrix.T, axis=1) 

if __name__=='__main__':
    A = numpy.random.rand(5,4)
    pca_trans_eig, pca_directions_eig, pca_variances_eig = pca_eig(A)
    pca_trans_svd, pca_directions_svd, pca_variances_svd = pca_svd(A)

    print pca_trans_eig.shape    
    print pca_trans_eig
    print pca_trans_svd.shape
    print pca_trans_svd
    print 
    print pca_directions_eig.shape
    print pca_directions_eig
    print pca_directions_svd.shape
    print pca_directions_svd
    print 
    print pca_variances_eig
    print pca_variances_svd
    
