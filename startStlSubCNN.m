% stlSubset for CNN  
clear all; close all; clc;
addpath(genpath('DeepLearnToolbox'));
addpath loadDataset;

%% load dataset
[trainData, trainLabel, testData, testLabel] = stlSubsetGenerateData();
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
    struct('type', 'c', 'outputmaps', 6,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 12,  'kernelsize', 3)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 8,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
};
opts.alpha = 1 ;
opts.batchsize = 100 ;     % needs to change according to train number
opts.numepochs = 100;      % long time  seconds per poches
opts.lowThreshold = 1e-6 ;
%%
fprintf('initiate cnn....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);
%%
 %load('dcm/cnn_128_6_12_5_500_mean');  % load cnn which has been trained
%%
fprintf('start training cnn...\n');
tic;
cnn = cnntrain(cnn, trainData, trainLabel, opts);
toc;
fprintf('cnn training completes\n');

% save('dcm/cnn128_6_16_5_100_j_mean', 'cnn', '-v7.3');
% disp('model saved-->dcm/cnn128_6_16_5_100_j_mean');

% load dcm/testData;
% load dcm/testLabel;
%% commence the cnn test
% load('dcm/cnn_128_6_16_5_1000_mean'); 
fprintf('cnn test commences :\n');
[ratio, er, bad] = cnntest(cnn, testData, testLabel);
fprintf('accuracy : %.2f %%\n', double(ratio * 100) );
fprintf('wrong number : %d / %d \n', numel(bad), size(testLabel, 2));
fprintf('cnn end !\n');


