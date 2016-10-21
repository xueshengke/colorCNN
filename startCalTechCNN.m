% stlSubset for CNN  
clear all; close all; clc;
addpath(genpath('DeepLearnToolbox'));

%% load dataset
[trainData, trainLabel, testData, testLabel] = calTech101GenerateData();
height = size(trainData, 1);
width = size(trainData, 2);
imageMap = size(trainData, 3);
trainNumber = size(trainData, 4);
testNumber = size(testData, 4);
classNumber = size(trainLabel, 1);

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
    struct('type', 'c', 'outputmaps', 10,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 8,  'kernelsize', 5)
    struct('type', 's', 'scale', 2)
};
opts.alpha = 1 ;
opts.batchsize = 100 ;     % needs to change according to train number
opts.numepochs = 1000;      % long time  seconds per poches
opts.lowThreshold = 1e-6 ;
%% initialize cnn 
fprintf('initiate cnn....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);

%% start training cnn
fprintf('commence training cnn ... \n');
tic ;
cnn = cnntrain(cnn, trainData, trainLabel, opts);
toc ;

%% start testing cnn
fprintf('commence testing cnn ... \n');
[ratio, error, bad] = cnntest(cnn, testData, testLabel);
fprintf('accuracy : %.2f %%\n', double(ratio * 100) );
fprintf('wrong number : %d / %d \n', numel(bad), size(testLabel, 2));
fprintf('cnn end !\n');

%% plot test error rate 
% plot(testErrorRate);
% grid on ;
% title('stl CNN');
% xlabel('epoch');
% ylabel('test error rate');


