% Main NN

%John_Schulz
%ECE465
%2/4/18

%% Initialization
clear all;
close all;
clc;
tic %start timer

%% Define hyperparameters for our network
%%=========================================================================
LearningRate = 0.1; % Skips made by gradient adjustment
noEpochs = 100001; % Number of iterations of training
plotRate = 101; % How often to test and display the network performance
%%=========================================================================

%% Test Data set - self explanatory
inputDataAND = [0 0 0
                0 1 0
                1 0 0
                1 1 1];

inputDataXOR = [0 0 0
                0 1 1
                1 0 1
                1 1 0];

inputDataOR =  [0 0 0
                0 1 1
                1 0 1
                1 1 1];
            
% pick one data set for this network            
inputData = inputDataOR; % assign any one of the data from above
%%=========================================================================
%% weights or parameters

    % Assign weights randomly at start but not zeros! why?
    % A baseline of values for weights must be made

w1 = 2*rand(3,3)-1; %weights layer 0 to 1
w2 = 2*rand(1,4)-1; %weights layer 1 to 2

%%=========================================================================

%% Start training the network
saveData = zeros(round(noEpochs/plotRate),6); %setup matrix to save data
k=1;
for n=1:noEpochs
    % run one epoch and update the weights 
    [w1,w2, J, h, y, delta1, delta2] = MSI_batchNN(inputData,w1,w2,LearningRate);
    
    % test the network ocassionally and save data
    if(mod(n,plotRate)==0)
        saveData(k,:) = [k J delta1' delta2];
        k=k+1;
        %----------------------------------------------------------------------
        % testing
        disp(['Epoch no:' num2str(n)  ', J=' num2str(J)])
        [w1,w2, J, h, y, delta1, delta2] = MSI_batchNN([0 0 0],w1,w2,LearningRate);
        disp(['[0 0] ->' num2str(h)]);
        [w1,w2, J, h, y, delta1, delta2] = MSI_batchNN([0 1 0],w1,w2,LearningRate);
        disp(['[0 1] ->' num2str(h)]);
        [w1,w2, J, h, y, delta1, delta2] = MSI_batchNN([1 0 0],w1,w2,LearningRate);
        disp(['[1 0] ->' num2str(h)]);
        [w1,w2, J, h, y, delta1, delta2] = MSI_batchNN([1 1 1],w1,w2,LearningRate);
        disp(['[1 1] ->' num2str(h)]);
    end    
end
%% =========================================================================
toc %end time measurement

%Plot [J] the Cost Function
figure(1)
    scatter(saveData(:,1),saveData(:,2)), grid on;
    xlabel('Iterations'), ylabel('Error');
    title('Calculated Cost');
    xlim([0 120]);
     
    
