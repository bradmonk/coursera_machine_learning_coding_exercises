%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
%% ================ Part 1: Feature Normalization ================

% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.


fprintf('Running gradient descent ...\n');


nloops = 400;
num_iters = 400;

ai = linspace(.001,.03,nloops);
Ji = zeros(num_iters, nloops);
Ti = zeros(3, nloops);

% Choose some alpha value
for nn = 1:nloops

    %alpha = .01;
    alpha = ai(nn);
    
    % Init Theta and Run Gradient Descent 
    theta = zeros(3, 1);
    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
    
    Ti(:,nn) = theta;
    Ji(:,nn) = J_history;

end


close all; clc;
fh1=figure('Units','normalized','OuterPosition',[.02 .1 .95 .8],'Color','w','MenuBar','none'); 
ax1 = axes('Position',[.05 .1 .40 .8],'Color','none');
    xlabel('Number of iterations');
    ylabel('Cost J');
    hold on;
ax2 = axes('Position',[.52 .1 .40 .8],'Color','none');
    xlabel('\theta_1'); 
    ylabel('\theta_2');
    zlabel('J');
    view(50,20); grid on;
    hold on;

axes(ax1) 
plot(  repmat([1:size(Ji,1)]',1,nloops)  , Ji, '-b', 'LineWidth', 2);
pause(.2)

axes(ax2)
surf(Ti(2,:), Ti(3,:), Ji')


% Display gradient descent's result
[MinJ , MinThetaInd] = min(min(Ji));

fprintf('\nLowest cost value found in test loop: %5.0f \n', MinThetaInd);
fprintf('Using alpha value: %13.3f \n', ai(MinThetaInd));

fprintf('\nTheta computed from gradient descent (on Z scaled data): \n')
fprintf('\n    %.3f ', Ti(:,MinThetaInd));
fprintf('\n\n')

Tnorm = pinv(X' * X) * X' * y;

fprintf('\nTheta computed from normal equation (on Z scaled data): \n')
fprintf('\n    %.3f ', Tnorm);
fprintf('\n\n')


theta = Ti(:,MinThetaInd);

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

x = ([1650 , 3] - mu) ./ sigma;

X = [1 x];

price = X * theta;

% ============================================================

fprintf(['\nPredicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n\n %4.s $%.2f\n\n\n'],' ', price);



%% ================ Part 3: Normal Equations ================

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('\nTheta computed from the normal equations (on raw data): \n');
fprintf(' %12.2f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

X = [1 , 1650 , 3];

price = X * theta;

% ============================================================

fprintf(['\nPredicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n\n %4.s $%.2f\n\n\n'],' ', price);

