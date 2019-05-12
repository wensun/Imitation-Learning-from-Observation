import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import scipy as si
from IPython import embed


def median_trick(X, scale = 1.):
    '''
    median trick for computing the sigma for RBF kernel, exp(-\|x-x'\|^2/sigma^2)
    '''
    N = X.shape[0]
    pdists = si.spatial.distance.pdist(X) #compute all pairwise euclidean distances. Warning: could be slow when N is large.
    return scale*np.median(pdists) #return the median as sigma.


def MMD_close_form_solution(X, alphas, sigma = None, x = None):
    '''
    X: n x d
    alphas: coefficients for each x in X
    
    Goal: max_{w, |w|_2\leq 1} w (sum_{i=1}^n phi(x_i)alpha_i)
    solution: f(x) = \sum_{i} \alpha_i k(x, x_i) / \sqrt{\sum_{i,j} \alpha_i\alpha_j k(x_i, x_j)}
    
    return: 1.f_xs: n-dim vector, each entry corresponds f(x_i), with x_i in X
    optionally,2. return f(x) for an arbitary input x
    '''

    N,d = X.shape
    if sigma is None: 
        sigma = median_trick(X)
    kernel_mat = rbf_kernel(X = X, gamma = 1./(sigma**2))

    kron_alphas = np.outer(alphas,alphas)
    kernel_met_alphas = kron_alphas*kernel_mat #for i,j'th entry, its alpha_i*alpha_j k(x_i, x_j)

    denominator = np.sqrt(np.sum(kernel_met_alphas)) #sum_{i,j} alpha_i alpha_j kernel(x_i,x_j)
    #print denominator
    f_xs = np.dot(kernel_mat, alphas)/denominator  # for each x_i in X, \sum_j alpha_j kernel(x_i, x_j) / demonimator

    if x is None:
        return f_xs, None
    else:
        kernel_vect = rbf_kernel(X = X, Y = x.reshape(1,d), gamma = 1./(sigma**2))
        return f_xs, np.dot(alphas, kernel_vect[:,0])/denominator, 



if __name__ == "__main__":

    dim = 20
    N = 200
    X1 = np.random.multivariate_normal(np.ones(dim)*0.001, np.eye(dim)*0.5, size = N)
    alpha1 = np.ones(N)
    X2 = np.random.multivariate_normal(np.zeros(dim), np.eye(dim)*0.5, size = N)
    alpha2 = -np.ones(N)

    X = np.vstack((X1, X2))
    alphas = np.hstack((alpha1, alpha2))

    print X.shape, alphas.shape

    fxs,_ = MMD_close_form_solution(X = X, alphas = alphas, x = None)
    mmd = np.sum(fxs.dot(alphas))/N

    print mmd



    




