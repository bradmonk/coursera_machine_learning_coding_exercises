function [J, G] = costFuncL3(X,y,T,L)

J = (1/length(y)) .* ( -y' * log(sigmoid(X*T)) - (1-y)' * log(1-sigmoid(X*T)) ) ...
    +  ((L / (2*length(y))) * sum(T(2:end).^2));


G = ( (1/length(y)) * X' * (sigmoid(X*T) - y) )  +  ((L / length(y)) * [0; T(2:end)]);

end