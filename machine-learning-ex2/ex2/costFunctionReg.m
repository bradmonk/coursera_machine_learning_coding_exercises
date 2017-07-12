function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

T = theta;

Tn = T(2:end);

h = sigmoid(X * T);

% keyboard

% COST FUNCTION

J = (1/m) .* ( -y' * log(h) - (1-y)' * log(1-h) ) +  ((lambda / (2*m)) * sum(Tn.^2));

% keyboard

% GRADIENT DESCENT (PARTIAL DERIVATIVE - ALPHA WILL BE TAKEN CARE OF LATER USING fminunc)
% keyboard

% grad(1) = (1/m) * X0' * ( sigmoid(X0 * T0) - y);

% grad(2:end) = ( (1/m) * Xn' * (sigmoid(Xn*Tn) - y) )  +  ((lambda / m) * Tn) ;  

grad = ( (1/m) * X' * (sigmoid(X*T) - y) )  +  [0; ((lambda / m) * Tn)] ;  

% =============================================================

end
