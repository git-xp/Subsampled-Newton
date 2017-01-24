function d = comp_logistic_diag(X, Y, w)
    d = 1./(1+exp(Y.*(X*w)))./(1+exp(-Y.*(X*w)));
%     d = exp(X*w)./((1+exp(X*w)).^2);
%     d = 1./(1+exp(-X*w))./(1 + exp(X*w));
end
