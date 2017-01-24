function l = logistic_loss(X,Y,w)
    %l = mean( log( 1./(1+exp(-Y.*(X*w)) ) ) );
    %l = -sum(log((1./(1+exp(-Y.*(X*w))))));
    l = sum(log(1 + exp(-Y.*(X*w))));
    
%     l = sum(log(1+exp(X*w)) - Y.*(X*w));
end
