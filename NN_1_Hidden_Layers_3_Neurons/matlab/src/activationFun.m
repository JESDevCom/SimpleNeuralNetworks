% activationFun
%John_Schulz
%ECE465
%2/4/18

function [z] = activationFun(x)

%Sigmoid
z = 1./(1+exp(x));

end