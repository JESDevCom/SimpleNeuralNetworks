% costFun - calculate cost function
%John_Schulz
%ECE465
%2/4/18

%error function
function [J] = costFun(y,h)

J = ((y * log(h)) + ((1 - y) * log(1 - h))) ;

end