function w = newton_solver(X,Y,niters,get_hessian,get_grad,lambda, w0)
% input:
%       X,Y     ---- data matrices
%       niter   ----- number of iterations
%       get_hessian     ---- function to compute the diagonal D^2
%       get_grad        ---- function to compute gradient
%       lambda          ---- ridge parameter
%       w0              ---- inital point
% output:
%       w               ---- optimal solution
%
% use this function to get optimal point
% it doesn't report solution errors
%
% written by Peng Xu, Jiyan Yang, 2/20/2016


    if (nargin < 6)
        lambda = 0;
    end

    [n,d] = size(X);
    w = w0;
    for i = 1:niters      
        D = get_hessian(X,Y,w);
        c = get_grad(X,Y,w);

        % solve the linear system using CG
        [v,~] = pcg(X'*bsxfun(@times,D,X) + lambda*eye(d), X'*c + lambda*w, 1e-6, 2000);
        w = w - v;
    end

end

