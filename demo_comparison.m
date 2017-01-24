%% test on real data sets ---- sketch_newton with full gradient
% In this demo, we train a ridge logistic regression model on libsvm
% 'adult' dataset, which can be downloaded here
%
%   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
%
% written by Peng Xu, 7/25/2016

clear all; clc; close all;
filename = 'adult';
[Y, X1] = libsvmread('a9a');% Need download the dataset first. See above.
X = full(X1(:,1:end-1)); % sparse representation slows down computation in this case.
X = [X, ones(size(X,1),1)]; % add 1 for bias term. 
% normalizing the columns
norms = sqrt(sum(X.^2));
X = bsxfun(@ldivide, norms, X);
[n, d] = size(X);


%%
% Problem
Prob.loss = @logistic_loss;
Prob.hessian = @comp_logistic_diag;
Prob.grad = @comp_logistic_grad;
lambda = .01;
Prob.lambda = lambda;

% find the solution first
w_opt= newton_solver(X,Y,30,Prob.hessian,Prob.grad,Prob.lambda, zeros(d,1));
l_opt = (Prob.loss(X,Y,w_opt) + lambda*(w_opt'*w_opt)/2)/n;
Prob.condition = cond(X'*bsxfun(@times,Prob.hessian(X,Y,w_opt),X) + lambda*eye(d));
H_opt = X'*bsxfun(@times,Prob.hessian(X,Y,w_opt),X);
sigs = svds(H_opt,d);
Prob.kappa = cond(X'*bsxfun(@times,Prob.hessian(X,Y,w_opt),X) + lambda*eye(d));
Prob.hatkappa = (n*max(Prob.hessian(X,Y,w_opt).*sum(X.*X,2)) + lambda)/(sigs(end) + lambda);
Prob.barkappa = (n*max(Prob.hessian(X,Y,w_opt).*sum(X.*X,2)) + lambda)/lambda;
Prob.w_opt = w_opt;
Prob.l_opt = l_opt;

Prob

Prob.w0 = zeros(d,1); % intial point
%%

% sampling sizes
uniform_sample_size = 200*d;
ls_sample_size = 20*d;
rns_sample_size = 20*d;

niters = 25; % number of iterations


methods{1} = struct('name','Newton','method','Newton','step_size',1,'niters',niters);
methods{2} = struct('name',sprintf('Uniform (%d)', uniform_sample_size), ...
    'method','Uniform', 'hessian_size', uniform_sample_size, 'step_size',1, 'niters', niters);
methods{3} = struct('name',sprintf('PLevSS (%d)', ls_sample_size), ...
    'method','LS', 'hessian_size',ls_sample_size,'mh', 10, 'step_size', 1, 'niters', niters);
methods{4} = struct('name',sprintf('RNormSS (%d)', rns_sample_size), ...
    'method','RNS', 'hessian_size', rns_sample_size, 'step_size', 1, 'niters', niters);
methods{5} = struct('name', sprintf('LBFGS-%d', 50), 'method','LBFGS',...
    'L',50, 'niters', niters*10, 'linesearch',true, 'alpha', 0.5, 'beta', 0.9);


% GD/AGD (optional)
% eta_gd = 0.8;
% eta_agd = 0.3;
% gamma = 0.9;
% methods{6} = struct('name','GD','method','GD','step_size',eta_gd,'niters',niters*30);
% methods{7} = struct('name','AGD','method','AGD','step_size',eta_agd,'gamma',gamma,'niters',niters*30);


results = cell(size(methods));
for i = 1:length(methods)
    fprintf('###################################\n Running %s ......\n###################################\n',methods{i}.name);
    [w, results{i}] = subsampled_newton(X,Y,Prob,methods{i});
end


fig_result;