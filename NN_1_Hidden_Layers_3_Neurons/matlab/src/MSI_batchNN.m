%MSI_batchNN
%John_Schulz
%ECE465
%2/4/18

function [w1_new,w2_new, J, h, y, delta1, delta2] = MSI_batchNN(inputData,w1,w2, LearningRate)

temp_delta1 =0;
temp_delta2 =0;


J =0;

for n=1:size(inputData,1)
    
    %take row 1 of input data and make it a column vector
    a0 = [1 inputData(n,1:2)]'; %input with bais value
    
    %% Forward propagation
    a1 = forwardProp(w1,a0);
    a2 = forwardProp(w2,[1;a1]);
    h = a2;

    
    %% Back propagation
        % Calculate error at the output layer
    y = inputData(n,3); %expected output
    
    %error in layer 2
    delta2 = h - y;
       
    % error in layer 1
    delta1t = ((w2' * delta2) .* ([1;a1].*(1-[1;a1])));
    delta1 = delta1t(2:end);
    
    % accumulate partial derivatives
    temp_delta2 = temp_delta2 + (delta2 * [1;a1]');
    temp_delta1 = temp_delta1 + (delta1 * a0'); 
    
    J = J + costFun(y,h);
end
J = J/-n;
% adjust weights
w2_new = w2 - (LearningRate * (temp_delta2/n));
w1_new = w1 - (LearningRate * (temp_delta1/n));

end

