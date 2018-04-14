function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

# Add col of ones to X representing the bias unit for each train example
# Why Adding bias unit at the end thr ouptut is diffent
X = [ones(m, 1) X];

# Calculate the hidden layer output (activations)
z1 = Theta1 * transpose(X);
a1 = sigmoid(z1);

# Add a row of ones representing the bias units
a1 = [ones(1, m); a1];

# Calculate the output layer activations
z2 = Theta2 * a1;
h = sigmoid(z2);

# Convert y to binary vectors
t = zeros(num_labels, m);  
for i = 1:m 
  t(y(i), i) = 1;
endfor
y = t;

# Calculating the cost
J = sum(sum(((-y .* log(h)) - (1 .- y) .* log(1 .- h)), 1)) * 1/m;

regularizationTerm = lambda / (2 * m) * ( ...
  sum(sum(Theta1(1:end,2:end) .* Theta1(1:end,2:end))) + ...
  sum(sum(Theta2(1:end,2:end) .* Theta2(1:end,2:end))) ...
  );

J = J + regularizationTerm;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta_layer1 = 0;
delta_layer2 = 0;

for i = 1:m
# Get the current activations of the current training example
a_layer2 = a1(1:end, i:i);
z_layer2 = z1(1:end, i:i);

a_layer3 = h(1:end, i:i);
z_layer3 = z2(1:end, i:i);

# Add the bias units
a_layer1 = transpose(X(i, 1:end));

# Get the last layer the output layer error (derivatives)
error_layer3 = a_layer3 - y(1:end, i);

# Get the error for the hidden layer and remove error of node 0(bias unit)
error_layer2 = (transpose(Theta2)(2:end, 1:end) * error_layer3) .* sigmoidGradient(z_layer2);

# Accumlate the gradients
delta_layer2 = delta_layer2 + error_layer3 * transpose(a_layer2);
delta_layer1 = delta_layer1 + error_layer2 * transpose(a_layer1);

# Get the unregularized gradient
Theta1_grad = 1.0 / m * delta_layer1;
Theta2_grad = 1.0 / m * delta_layer2;

endfor

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
