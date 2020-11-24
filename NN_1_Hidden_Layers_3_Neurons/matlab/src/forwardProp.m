%% Vectorized forward propagation

%John_Schulz
%ECE465
%2/4/18

% a = sxm matrix where: 
%                       s=number of units in layer l
%                       m=number of training examples
%
% b = bias input sxm matrix, obtained by repmat(b,1,m)
%      that is taking column vector of b1 and stacking m copies in columns
%       [               ]
%       [b1 b1 b1 ....b1]
%       [               ]


function [a] = forwardProp(Wc,x)

z = Wc*x; %multiply weights by their nodes
a = activationFun(-z);   %apply sigmoid function