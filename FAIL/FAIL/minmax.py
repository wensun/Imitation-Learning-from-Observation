import numpy as np
from generate_traj import traj_segment_generator
from mmd import MMD_close_form_solution, median_trick
from IPython import embed
from sklearn.decomposition import PCA

def evaluate_likelihood_ratio(X, A, pi, A_ref_logps):
    #for each (x,a) pair, compute the likelihood ration: pi(a|x) / \pi_{ref}(a|x)
    #where log pi_{ref}(a|x) is stored in A_ref_logps. 
    #X, A, A_ref_probs are in list format
    assert len(A_ref_logps) == X.shape[0]

    #logps = pi.logps(np.array(A), np.array(X))
    logps = pi.logps(X, A)
    return np.exp(logps - A_ref_logps)  #pi(a|x)/pi_ref(a|x)


def game_utility(ratios, f_XXstar_next):
    #ratios: pi_n(a|x)/pi_ref(a|x)
    #f_XXstar: f_max(x) + f_{max}(\tilde{x})

    N = ratios.shape[0] #N size of (x,a)
    assert f_XXstar_next.ndim == 1
    assert f_XXstar_next.shape[0] > N
    N_star = f_XXstar_next.shape[0] - N

    f_X_next = f_XXstar_next[0:N]
    f_Xstar_next = f_XXstar_next[N:] #expert data
    game_u = ratios.dot(f_X_next)/N - np.sum(f_Xstar_next)/N_star
    return game_u

def subset_feature_select(X, h = None):
    #return X
    #PCA:
    pca = PCA(n_components = X.shape[1])
    return pca.fit_transform(X)

    #return X#[:, 3:]
    #return X[:,-15:]
    #if h is None:
    #    return X
    #else:
    ##    X[:,3:6] *= ((h+1)**0.5)
    #    return X


def minmax_solver(X, A, X_next, X_star_next, A_ref_logps, pi, T, lr = 1e-3, h = None, diag = False):
    #run no-regret update T iterations
    # X, A, X_next: (x,a, x') pair
    #A_ref_logps: the log-likelihood of pi_{ref}(a|x)
    #pi: the policy that needs to be updated
    #X_star_next: (x) from expert policy
    #T: number of iterations to run

    #median_sigma = median_trick(X_star_next, scale = 1.)
    #print(median_sigma)
    N_star = X_star_next.shape[0]
    N = X.shape[0]
    assert N == A.shape[0]
    assert N == A_ref_logps.shape[0]

    XXstar_next = np.concatenate((X_next, X_star_next), axis = 0)

    pca = PCA(n_components = XXstar_next.shape[1])
    XXstar_next_pca = XXstar_next[:,0:9]# pca.fit_transform(XXstar_next)

    median_sigma = None
    midian_inv_diag = None
    min_utility = np.inf
    variables_min = None
    utilities = []

    #sigma = median_trick(subset_feature_select(X_star_next))/5.
    #sigma = median_trick(subset_feature_select(XXstar_next))
    sigma = median_trick(XXstar_next_pca)/5.
    #sigma = None

    for i in range(T):
        #compute ratios: pi(a|x) / pi_{ref}(a|x)
        pi_ratios = evaluate_likelihood_ratio(X, A, pi, A_ref_logps)/N
        assert pi_ratios.ndim == 1
        assert pi_ratios.shape[0] == N

        #call MMD:
        all_ratios = np.concatenate((pi_ratios, -np.ones(N_star)/N_star), axis = 0)  #size: N + N_star
        if i == 0:
            f_XXstar_next, _, inv_diag_cov, sigma,ret = MMD_close_form_solution(XXstar_next_pca,
                                                                                all_ratios,inv_diag_cov = None, sigma=sigma, x = None, diag = diag)
            median_sigma = sigma
            midian_inv_diag = inv_diag_cov
        else:
            f_XXstar_next,_,_,_,ret = MMD_close_form_solution(XXstar_next_pca, all_ratios, inv_diag_cov = midian_inv_diag,
                                                            sigma= median_sigma, x = None, diag=diag)

        #after compute f_max, we evaluate the game value:
        utility = all_ratios.dot(f_XXstar_next)
        assert ret == utility
        #utility = game_utility(pi_ratios, f_XXstar_next)
        if (i+1) % 10 == 0:
            print("In min-max game, at iter {0}, the game utility is {1}".format(i, utility))

        utilities.append(utility)
        if utility <= min_utility:
            min_utility = utility
            variables_min = pi.get_traniable_variables_flat()

        #update on policy now:
        f_X_next = f_XXstar_next[0:N] + 1e-7*np.sum(A*A,axis=1)
        f_over_pi_ref= f_X_next / (np.exp(A_ref_logps)+1e-7)  #f_max(\tilde{x})
        pi.update_policy(X, A, f_over_pi_ref, step_size = lr)

    #update the policy to the min utility version:
    pi.set_trainable_variable_flat(variables_min)

    return utilities




