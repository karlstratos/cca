from numpy.random import randn
from scipy.linalg import qr
from scipy.linalg import svd
from scipy.sparse import csc_matrix
from numpy import allclose, outer
from sparsesvd import sparsesvd

def mysparsesvd(sparsematrix, m):
    Ut, S, Vt = sparsesvd(sparsematrix, m)
    return Ut.T, S, Vt.T

extra_dim = 50
power_num = 5

def randsvd(M, m):
    Z = M * randn(M.shape[1], m + extra_dim) 
    Q, _ = qr(Z, mode='economic')
    for _ in range(power_num):
        Z = M.T * Q
        Q, _ = qr(Z, mode='economic')
        Z = M * Q 
        Q, _ = qr(Z, mode='economic')
    
    U_low, svals, Vt = svd(Q.T * M, full_matrices=False)
    
    svals = svals[:m]
    U = csc_matrix(Q) * U_low[:,:m] # bring U back to the original dimension
    V= Vt.T[:, :m]
    
    return U, svals, V

def randsvd_centered(M, v1, v2, m):
    T = randn(M.shape[1], m + extra_dim) 
    Z = M * T - v1 * (v2.T * T)            
    Q, _ = qr(Z, mode='economic')
    for _ in range(power_num):
        Z = M.T * Q - v2 * (v1.T * Q)
        Q, _ = qr(Z, mode='economic')
        Z = M * Q - v1 * (v2.T * Q)
        Q, _ = qr(Z, mode='economic')

    B = Q.T * M - (Q.T * v1) * v2.T
    U_low, svals, Vt = svd(B, full_matrices=False)
    
    svals = svals[:m]
    U = csc_matrix(Q) * U_low[:,:m] # bring U back to the original dimension
    V= Vt.T[:, :m]
    
    return U, svals, V

if __name__=='__main__':
    M = randn(100,300)
    v1 = randn(100,1)
    v2 = randn(300,1)
    m = 60
    
    O = M - outer(v1, v2)
    U_svd, svals_svd, Vt_svd = svd(O)
    U_svd = U_svd[:,:m]
    svals_svd = svals_svd[:m]
    Vt_svd = Vt_svd[:m,:]
    
    U_approx1, svals_approx1, Vt_approx1 = randsvd(csc_matrix(O), m)
    U_approx2, svals_approx2, Vt_approx2 = randsvd_centered(csc_matrix(M), csc_matrix(v1), csc_matrix(v2), m)
    
    assert(allclose(svals_svd, svals_approx1) and allclose(svals_svd, svals_approx2))

