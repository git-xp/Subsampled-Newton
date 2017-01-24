
import os
import numpy as np
import numpy.linalg as la
from scipy import sparse
import time
import copy

def newton_solver(X,Y,problem, niters):
    loss = problem.loss
    get_grad = problem.grad
    get_hessian = problem.hessian
    alpha = problem.alpha
    n,d = X.shape
    
    eta = 1
    w = np.zeros(d)
    
    t = []
    sols = []
    results = {}

    print("Newton solver starts ......")
    tic = time.time()
    t.append(0)
    sols.append(np.zeros(d))
    for i in range(niters):
        # compute the exact hessian
        D2 = get_hessian(X,Y,w)
        H = np.multiply(X.T, D2).dot(X) + alpha * np.eye(d)
        # compute the full gradient
        c = get_grad(X,Y,w)
        g = X.T.dot(c) + alpha * w
        # compute the search direction
        z = np.linalg.solve(H, -g)
        # update 
        w += eta * z
        # print i, np.sum(z**2), loss(X,Y,w)
        t.append(time.time() - tic)
        sols.append(copy.deepcopy(w))
    sol = w
    print("Newton solver ends")



    print("Further postprocessing ......")
    l = [loss(X,Y,w) for w in sols]
    
    results['t'] = t
    results['l'] = l
    if not problem.w_opt is None:
        w_opt = problem.w_opt
        errs = sols - w_opt
        errs = [np.linalg.norm(err) for err in errs]
        errs = errs/np.linalg.norm(w_opt)
        results['errs'] = errs
    print("Done! :)")
    return (sol,results)

def comp_apprx_ridge_lev(X,alpha,r,D=None):
    n,d = X.shape
    rn1 = np.random.randint(r,size=n)
    rn2 = np.random.randint(r,size=d)
    if D != None:
        S1 = sparse.coo_matrix(((np.random.randint(2,size=n)*2-1)*D,(rn1, np.arange(n))), shape=(r,n))
    else:
        S1 = sparse.coo_matrix(((np.random.randint(2,size=n)*2-1),(rn1, np.arange(n))), shape=(r,n))
    S2 = sparse.coo_matrix(((np.random.randint(2,size=d)*2-1)*np.sqrt(alpha),(rn2, np.arange(d))), shape=(r,d))
    SDX = S1.dot(X) + S2
    _, R = la.qr(SDX)
    invRG = la.solve(R,(np.random.randn(d, int(d/2))/np.sqrt(int(d/2))))
    if D != None:
        lev = D*D*np.sum((X.dot(invRG))**2,axis=1)
    else:
        lev = np.sum((X.dot(invRG))**2,axis=1)
    return lev

    
def subsampled_newton_lev(X,Y,problem,params):
    """
    subsampled newton solver with full gradient for the following problem:
        min_w sum_i loss(w' * x_i, y_i) + alpha * norm(w)^2
            
    Input: 
        X,Y                     ---- data matrices
        problem:
            problem.loss        ---- loss, get_grad, get_hessian
            problem.grad        ---- function to compute gradient
            problem.hessian     ---- function to compute the diagonal D^2
            problem.alpha       ---- ridge parameters
            problem.w0          ---- initial start point (optional)
            problem.w_opt       ---- optimal solution (optinal)
            problem.l_opt       ---- minimum loss (optional)
            problem.condition   ---- condition number (optional)
        params:
            params.method       ---- hessian sketch schemes
            params.hessian_size ---- sketching size for hessian
            params.step_size    ---- step size
            params.mh           ---- Hessian approximation frequency 
 
     output:
        sol ---- final solution
        results:
            results.t       ---- running time at every iteration
            results.err     ---- solution error (if problem.w_opt given)
            results.l       ---- objective value 
            results.sol     ---- solution at every iteration
            
    """

    loss = problem.loss
    get_grad = problem.grad
    get_hessian = problem.hessian
    alpha = problem.alpha
    n,d = X.shape
    
    niters = params['niters']
    eta = 1
    w = params['w0']
    mh = params['mh']
    hessian_size = params['hessian_size']
    # method = params['method']
    r = min(10000,20*d);

    t = []
    sols = []
    results = {}

    print("subsample Newton (PlevSS) solver starts ......")
    tic = time.time()
    t.append(0)
    sols.append(copy.deepcopy(w))
    for i in range(niters):
        # every mh iteration recompute the approximate leverage scores
        if i % mh == 0: 
            D2 = get_hessian(X,Y,w)
            lev = comp_apprx_ridge_lev(X,alpha,r,np.sqrt(D2))
            p0 = lev/np.sum(lev)
            q = p0*hessian_size
            q[q > 1] = 1

        # compute the subsample Hessians
        idx = np.random.rand(n) < q 
        p_sub = q[idx]
        X_sub = X[idx,:] 
        D2_sub = get_hessian(X_sub,Y[idx],w)
        H = X_sub.T.dot(X_sub * ((D2_sub/p_sub)[:,np.newaxis])) + alpha * np.eye(d); 

        # compute the full gradient
        c = get_grad(X,Y,w)
        g = X.T.dot(c) + alpha * w

        # compute the search direction
        # TODO: FIGURE OUT BEST SOLVER IN PYTHON
        z = np.linalg.solve(H, -g)

        # update
        w += eta * z

        t.append(time.time() - tic)
        sols.append(copy.deepcopy(w))
    sol = w
    print("subsample Newton (PlevSS) solver ends")


    
    print("Further postprocessing ......")
    l = [loss(X,Y,w) for w in sols]
    results['t'] = t
    results['l'] = l
    if not problem.w_opt is None:
        w_opt = problem.w_opt
        errs = sols - w_opt
        errs = [np.linalg.norm(err) for err in errs]
        errs = errs/np.linalg.norm(w_opt)
        results['errs'] = errs
    print("Done! :)")
    return (sol,results)


def subsampled_newton_rns(X,Y,problem,params):
    loss = problem.loss
    get_grad = problem.grad
    get_hessian = problem.hessian
    alpha = problem.alpha
    n,d = X.shape
    
    niters = params['niters']
    eta = 1
    w = params['w0']
    mh = params['mh']
    hessian_size = params['hessian_size']
    # method = parms['method']


    t = []
    sols = []
    results = {}

    print("subsample Newton (RnormSS) solver starts ......")
    tic = time.time()
    rnorms = np.sum(X**2, axis=1)
    t.append(0)
    sols.append(copy.deepcopy(w))
    for i in range(niters):
        # compute the row norm squares
        D2 = get_hessian(X,Y,w)
        p = D2*rnorms
        p = p/np.sum(p)
        q = p*hessian_size
        q[q>1] = 1

        # compute the subsample Hessians
        idx = np.random.rand(n) < q 
        p_sub = q[idx]
        X_sub = X[idx,:] 
        D2_sub = get_hessian(X_sub,Y[idx],w)
        H = X_sub.T.dot(X_sub * ((D2_sub/p_sub)[:,np.newaxis])) + alpha * np.eye(d); 

        # compute the full gradient
        c = get_grad(X,Y,w)
        g = X.T.dot(c) + alpha * w

        # compute the search direction
        # TODO: FIGURE OUT BEST SOLVER IN PYTHON
        z = np.linalg.solve(H, -g)

        # update
        w += eta * z

        t.append(time.time() - tic)
        sols.append(copy.deepcopy(w))
    sol = w
    print("subsample Newton (RnormSS) solver ends")


    
    print("Further postprocessing ......")
    l = [loss(X,Y,w) for w in sols]
    results['t'] = t
    results['l'] = l
    if not problem.w_opt is None:
        w_opt = problem.w_opt
        errs = sols - w_opt
        errs = [np.linalg.norm(err) for err in errs]
        errs = errs/np.linalg.norm(w_opt)
        results['errs'] = errs
    print("Done! :)")
    return (sol,results)

def subsampled_newton_uniform(X,Y,problem,params):
    loss = problem.loss
    get_grad = problem.grad
    get_hessian = problem.hessian
    alpha = problem.alpha
    n,d = X.shape
    
    niters = params['niters']
    eta = 1
    w = params['w0']
    mh = params['mh']
    hessian_size = params['hessian_size']
    # method = parms['method']


    t = []
    sols = []
    results = {}

    print("subsample Newton (Uniform) solver starts ......")
    tic = time.time()
    t.append(0)
    sols.append(copy.deepcopy(w))
    for i in range(niters):

        # compute the subsample Hessians
        idx = np.random.choice(n,hessian_size)
        p_sub = 1.0/n
        X_sub = X[idx,:] 
        D2_sub = get_hessian(X_sub,Y[idx],w)
        H = X_sub.T.dot(X_sub * ((D2_sub/p_sub)[:,np.newaxis])) + alpha * np.eye(d); 

        # compute the full gradient
        c = get_grad(X,Y,w)
        g = X.T.dot(c) + alpha * w

        # compute the search direction
        # TODO: FIGURE OUT BEST SOLVER IN PYTHON
        z = np.linalg.solve(H, -g)

        # update
        w += eta * z

        t.append(time.time() - tic)
        sols.append(copy.deepcopy(w))
    sol = w
    print("subsample Newton (Uniform) solver ends")


    
    print("Further postprocessing ......")
    l = [loss(X,Y,w) for w in sols]
    results['t'] = t
    results['l'] = l
    if not problem.w_opt is None:
        w_opt = problem.w_opt
        errs = sols - w_opt
        errs = [np.linalg.norm(err) for err in errs]
        errs = errs/np.linalg.norm(w_opt)
        results['errs'] = errs
    print("Done! :)")
    return (sol,results)