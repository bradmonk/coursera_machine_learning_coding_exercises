function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta


T = theta;

h = sigmoid(X * T);

% COST FUNCTION

J = (1/m) .* ( -y' * log(h) - (1-y)' * log(1-h) );

% GRADIENT DESCENT

grad = (1/m) * X' * (sigmoid(X*T) - y);  % PARTIAL DERIVATIVE (ALPHA WILL BE TAKEN CARE OF LATER USING fminunc)




% T = T - (a/m) * X' * (sigmoid(X*T) - y);
% gradient = zeros(2,1);
% gradient(1) = 2*(theta(1) - 5);
% gradient(2) = 2*(theta(2) - 5);



% =============================================================

end
