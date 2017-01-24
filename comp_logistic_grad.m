function c = comp_logistic_grad(X, Y, w)
    c = -Y./(1+exp(Y.*(X*w)));
%     c = 1./(1+exp(-X*w)) - Y;
end