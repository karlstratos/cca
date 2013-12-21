import argparse
from numpy import dot
from numpy import cov
from numpy import diag
from numpy import mean
from numpy import allclose
from numpy.linalg import eig
from numpy.random import rand
from io import say
from io import read_embeddings
from io import write_embeddings
from scipy.linalg import svd

def pca_eig(A):
    M = center_cols(A)
    pca_directions, pca_variances = sorted_eig(cov(M.T))
    pca_trans = dot(M, pca_directions)
    return pca_trans, pca_directions, pca_variances

def pca_svd(A):
    M = center_cols(A)
    U, S, VT = svd(M, full_matrices=False)
    pca_trans = dot(U, diag(S))
    pca_directions = VT.T
    pca_variances = pow(S,2)/(A.shape[0]-1)     
    return pca_trans, pca_directions, pca_variances

def sorted_eig(matrix):
    [evals,evecs] = eig(matrix)
    idx = evals.argsort()[::-1]   
    evecs = evecs[:,idx] 
    evals = evals[idx]
    return evecs, evals

def center_cols(matrix):
    return matrix - mean(matrix.T, axis=1) 

def perform_pca(embedding_file, pca_dim):
    freqs, words, _, _, _, A = read_embeddings(embedding_file)
    say('performing PCA to reduce dimensions from {} to {}'.format(A.shape[1], pca_dim))            
    pca_trans, _, _ = pca_svd(A) 
    A_pca = pca_trans[:,:pca_dim]
    write_embeddings(freqs, words, A_pca, embedding_file + '.pca' + str(pca_dim))

if __name__=='__main__':
    argparser = argparse.ArgumentParser('Performs PCA on embeddings')
    argparser.add_argument('--embedding_file', type=str, help='file containing embeddings')
    argparser.add_argument('--pca_dim', type=int, default=100, help='reduce dimension to this number')
    argparser.add_argument('--debug', action='store_true', help='check the correctness of the PCA methods')
    args = argparser.parse_args()

    if args.embedding_file:
        perform_pca(args.embedding_file, args.pca_dim) 

    if args.debug:    
        A = rand(5,4)
        pca_trans_eig, pca_directions_eig, pca_variances_eig = pca_eig(A)
        pca_trans_svd, pca_directions_svd, pca_variances_svd = pca_svd(A)

        assert(allclose(abs(pca_trans_eig), abs(pca_trans_svd)))
        assert(allclose(abs(pca_directions_eig), abs(pca_directions_svd))) 
        assert(allclose(abs(pca_variances_eig), abs(pca_variances_svd)))
