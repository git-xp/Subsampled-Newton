function [sol, results] = subsampled_newton(X,Y,problem,params)
% subsample newton solver with full gradient for the following problem:
%       min_w sum_i l(w'*x_i,y_i) + lambda*norm(w)^2
% In this solver, we also add LBFGS, GD,AGD methods for comparison.
%
% input: 
%       X,Y                     ---- data matrices
%       problem:
%           problem.loss        ---- loss, get_grad, get_hessian
%           problem.grad        ---- function to compute gradient
%           problem.hessian     ---- function to compute the diagonal D^2
%           problem.lambda      ---- ridge parameters
%           problem.w0          ---- initial start point (optional)
%           problem.w_opt       ---- optimal solution (optinal)
%           problem.l_opt       ---- minimum loss (optional)
%           problem.condition   ---- condition number (optional)
%       params:
%           params.method       ---- hessian sketch schemes
%           params.hessian_size ---- sketching size for hessian
%           params.step_size    ---- step size
%           params.mh           ---- Hessian approximation frequency 
% 
% output:
%       sol ---- final solution
%       results:
%           results.t       ---- running time at every iteration
%           results.err     ---- solution error (if problem.w_opt given)
%           results.l       ---- objective value 
%           results.sol     ---- solution at every iteration
% 
% 
%
% written by Peng Xu, Jiyan Yang, 2/20/2016


    
    if nargin == 3
        params = 0;
    end

    loss = problem.loss;
    get_grad = problem.grad;
    get_hessian = problem.hessian;

    [n,d] = size(X);
    
    % default setting
    lambda = 0;
    method= 'ssn';
    niters = 25;            % total number of iterations
    r = min(10000,20*d);    % parameter for computing approximate leverage scores
    m1 = 1;                 % number of iterations for updating stale leverage scores
    linesearch = false;
    eta = 1;                % step size
    w0 = zeros(d,1);        % initial point

    % check params
    if isfield(problem, 'lambda')
        lambda = problem.lambda;
    end
    
    if isfield(problem, 'w0')
        w0 = problem.w0;
    end

    if isfield(params, 'method')
        method = params.method;
    end
    
    if isfield(params, 'hessian_size')
        s = params.hessian_size;
    end

    if isfield(params, 'step_size')
        eta = params.step_size;
    end

    if isfield(params,'niters')
        niters = params.niters;
    end
    
    if isfield(params, 'gamma')
        gamma = params.gamma;
    end
    
    if isfield(params,'r')
        r = params.r;
    end
    
    if isfield(params, 'mh')
        m1 = params.mh;
    end
    
    if isfield(params,'linesearch')
        linesearch = params.linesearch;
    end
        
    w = w0;
    t = zeros(niters,1);
    l = zeros(niters,1);
    sol = zeros(d, niters);
    
    % algorithm start
    fprintf('algorithm start ......\n');
    tic;
%     if strcmp(method, 'LS1')
%         lev0 = comp_apprx_ridge_lev(X,lambda,r);
%     end

    if strcmp(method, 'RNS')
        rnorms = sum(X.^2,2);
    end
    if strcmp(method, 'SGD')
        schedulers = randsample(n,niters,'true');
    end
    for i = 1:niters
        
        % compute hessian
        switch method
            case 'Newton'
                D2 = get_hessian(X,Y,w);
                H = X'*bsxfun(@times, D2, X) + lambda*eye(d);
            %case 'LS1'
            %    % using native reweighting scheme
            %    if mod(i-1,m1) == 0
            %        D2 = get_hessian(X,Y,w); 
            %    D2 = get_hessian(X,Y,w);
            %    lev = compute_exact_leverage_scores(bsxfun(@times,sqrt(D2),X),lambda);
            %    p = lev/sum(lev); 
            %    q = min(1,p*s);
            %    end
            %    idx = rand(n,1)<q;p_sub = q(idx);
            %    X_sub = X(idx,:); D2_sub = D2(idx);
            %    H = X_sub'*bsxfun(@times,D2_sub./p_sub,X_sub)+lambda*eye(d);

            case 'LS'
                % re-approximating scores every m1 iterations but never reweighting
                if mod(i-1,m1) == 0
                    D2 = get_hessian(X,Y,w);
                    lev = comp_apprx_ridge_lev(X,lambda,r,sqrt(D2));
                    p0 = lev/sum(lev);
                    q = min(1,p0*s);
                end
                idx = rand(n,1)<q; p_sub = q(idx);
                X_sub = X(idx,:); D2_sub = get_hessian(X_sub,Y(idx),w);
                H = X_sub'*bsxfun(@times,D2_sub./p_sub,X_sub)+lambda*eye(d);             
            
            case 'RNS'    
                D2 = get_hessian(X,Y,w);
                p = D2.*rnorms;
                p = p/sum(p);              
                q = min(1,p*s);idx = rand(n,1)<q;p_sub = q(idx);
                X_sub = X(idx,:); D2_sub = D2(idx);
                H = X_sub'*bsxfun(@times,D2_sub./p_sub,X_sub)+lambda*eye(d);
                
            case 'Uniform'
                idx = randsample(n,s); % default is sampling without replacement
                X_sub = X(idx,:);
                D2_sub = get_hessian(X_sub,Y(idx),w); 
                H = X_sub'*bsxfun(@times,D2_sub*n,X_sub)/s+lambda*eye(d);
        end
        
        
        
        if strcmp(method, 'SGD')
            datax = X(schedulers(i),:);
            z = -(datax'*(get_grad(datax,Y(schedulers(i)),w)) + lambda*w/n);
            eta = params.step_size/(1+i/5000);
            w = w + eta*z;
            
        elseif strcmp(method,'AGD')
            c = get_grad(X,Y,w);
            v = X'*c + lambda*w;
            if i == 1
                ys = w;
            end
            ys1 = w - eta*v;
            w = (1+gamma)*ys1 - gamma*ys;
            ys = ys1;
            
        else
            c = get_grad(X,Y,w);
            v = X'*c + lambda*w;
            switch method
                case 'LBFGS'
                    if i == 1
                        H = eye(d);
                        z = -v;
                        y = zeros(d,0);
                        S = zeros(d,0);
                    else
                        s_prev = alpha_prev*z_prev;
                        y_prev = v - v_prev;

                        if size(S,2) >=  params.L;
                            S = [S(:,2:end),s_prev];
                            y = [y(:,2:end),y_prev];
                        else
                            S = [S, s_prev];
                            y = [y, y_prev];
                        end
                        z = -lbfgs(v,S,y,speye(d));

                    end
                    z_prev = z;
                    v_prev = v;

                case 'GD'
                    z = -v;
                case {'LS','RNS','Uniform', 'Newton','LS1'}
                    [z,~] = pcg(H, -v, 1e-6, 1000); % suppress the output of pcg;  
            end
            
            if linesearch
                eta = 1;
                l = loss(X,Y,w);
                delta = params.alpha * z' * v;
                while (loss(X,Y,w + eta*z) >= l + delta * eta)
                    eta = eta * params.beta;
                end
                if eta == 1;
                    linesearch = false;
                end
            end
            alpha_prev = eta;
            w = w + eta*z;
        end

        sol(:,i) = w;
        t(i) = toc;
    end
    
    fprintf('main algorithm end\n');
    % better improve this using vector operations
    fprintf('Further postprocessing......\n')
    t = [0;t];
    sol = [w0,sol];
    results.t = t;
    results.sol = sol;
    for i = 1:niters+1
        w = sol(:,i);
        l(i) = (loss(X,Y,w) + lambda*(w'*w)/2)/n;
    end
    results.l = l;
    
    if isfield(problem, 'w_opt')
        w_opt = problem.w_opt;
        err = bsxfun(@minus, sol, w_opt);
        err = sqrt(sum(err.*err))';
        results.err = err/norm(w_opt,2);
    end
   
    if isfield(params,'name')
        results.name = params.name;
    end
    fprintf('DONE! :) \n');
end

function lev = comp_apprx_ridge_lev(X,lambda,r,D)
    % there might be a better implementation for the sparse transform
    if nargin < 4
        D = [];
    end
    [n,d] = size(X);
    rn1 = randi(r, [n,1]);
    rn2 = randi(r, [d,1]);
    if D
        S1 = sparse(rn1, 1:n, (randi(2,[n,1])*2-3).*D, r, n);
    else
        S1 = sparse(rn1, 1:n, randi(2,[n,1])*2-3, r, n);
    end
    S2 = sparse(rn2, 1:d, sqrt(lambda)*(randi(2,[d,1])*2-3), r, d);
    SDX = S1*X + S2;
    [~,R] = qr(SDX,0);
    invRG = R\(randn(d, floor(d/2))/sqrt(floor(d/2)));
    if D
        lev = D.^2.*sum((X*invRG).^2,2);
    else
        lev = sum((X*invRG).^2,2);
    end
end
