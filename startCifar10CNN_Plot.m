% stlSubset for CNN  
clear all; close all; clc;
addpath(genpath('DeepLearnToolbox'));
addpath('loadDataset');
%% load dataset
[trainData, trainLabel, testData, testLabel] = cifar10GenerateData();
height = size(trainData, 1);
width = size(trainData, 2);
imageMap = size(trainData, 3);
trainNumber = size(trainData, 4);
testNumber = size(testData, 4);
classNumber = size(trainLabel, 1);

% trainData = reshape(trainData, height * width, trainNumber);
% testData  = reshape(testData,  height * width, testNumber);
% %% mean patch to zero, two method have same effect
% % method 1
% trainData = trainData - repmat(mean(trainData, 2), 1, size(trainData, 2));
% testData = testData - repmat(mean(testData, 2), 1, size(testData, 2));
% % method 2
% % trainData = bsxfun(@minus, trainData, mean(trainData, 2));
% % testData = bsxfun(@minus, testData, mean(testData, 2));
% 
% trainData = reshape(trainData, height, width, trainNumber);
% testData  = reshape(testData,  height, width, testNumber);

fprintf('prepare trainData %d * %d * %d * %d \n', height, width, imageMap, trainNumber);
fprintf('prepare trainLabel %d * %d \n', size(trainLabel, 1), size(trainLabel, 2));
fprintf('prepare testData  %d * %d * %d * %d \n', height, width, imageMap, testNumber);
fprintf('prepare testLabel  %d * %d \n', size(testLabel, 1), size(testLabel, 2));

%% CNN design
% rand('state',0);
cnn.inputmaps = imageMap;         % gray = 1, color(RGB) = 3
cnn.classNum = classNumber;
cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 12,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 24,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
};
opts.alpha = 1 ;
opts.batchsize = 100 ;     % needs to change according to train number
opts.numepochs = 1 ;      % long time  seconds per poches
opts.lowThreshold = 1e-6 ;
%%
fprintf('initiate cnn....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);

iterNumber = 500;
testErrorRate = zeros(iterNumber, 1);
runningTime = zeros(iterNumber, 1);
for i = 1 : iterNumber
    %% start training cnn network
    fprintf('commence training cnn ... \n');
    tic ;
    cnn = relucnntrain(cnn, trainData, trainLabel, opts);
    loopTime = toc ;
    if i == 1
        runningTime(i) = loopTime;
    else
        runningTime(i) = runningTime(i -1) + loopTime;
    end
    %% start testing cnn network
    fprintf('commence testing cnn ... \n');
    [ratio, error, bad] = relucnntest(cnn, testData, testLabel);
    fprintf('%d / %d, Accuracy %.2f %%\n', i, iterNumber, ratio * 100) ;
    testErrorRate(i) = error  ;
end
displayAccuracy(testLabel, bad);

%% plot test error rate 
plot(testErrorRate);
grid on ;
title('cifar10 CNN');
xlabel('epoch');
ylabel('test error rate');


