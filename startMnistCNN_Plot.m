%% CNN application in minist data
clear all;clc;
addpath(genpath('DeepLearnToolbox'));
load mnist_uint8;

%% reconstruct data and normalize
imageSize = 28;
trainNumber = 60000;
testNumber = 10000;

trainData = double(reshape(train_x',imageSize,imageSize,trainNumber)) / 255;
trainLabel = double(train_y');
testData = double(reshape(test_x',imageSize,imageSize, testNumber)) / 255;
testLabel = double(test_y');
clear train_x train_y test_x test_y;

fprintf('prepare trainData %d * %d * %d \n', size(trainData, 1), size(trainData, 2), size(trainData, 3));
fprintf('prepare trainLabel %d * %d \n', size(trainLabel, 1), size(trainLabel, 2));
fprintf('prepare testData %d * %d * %d \n', size(testData, 1), size(testData, 2), size(testData, 3));
fprintf('prepare testLabel %d * %d \n', size(testLabel, 1), size(testLabel, 2));

%% Train a 6c-2s-12c-2s convolutional neural network 
% rand('state',0);
cnn.inputmaps = 1;         % gray image
cnn.classNum = size(trainLabel, 1);
cnn.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)   %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)  %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};
opts.alpha = 1 ;
opts.batchsize = 50 ;       % select batch from train data
opts.numepochs = 1 ;        % do not require too large value
opts.lowThreshold = 1e-6 ;

%% initiate cnn network
fprintf('commence inititate cnn ... \n');
cnn = cnnsetup(cnn, trainData, trainLabel);

iterNumber = 20;
testErrorRate = zeros(iterNumber, 1);
for i = 1 : iterNumber
    %% start training cnn network
    fprintf('commence training cnn ... \n');
    tic ;
    cnn = cnntrain(cnn, trainData, trainLabel, opts);
    toc ;
    %% start test cnn network
    fprintf('commence testing cnn ... \n');
    [ratio, error, bad] = cnntest(cnn, testData, testLabel);
    fprintf('Accuracy %.2f %%\n', ratio * 100) ;
    testErrorRate(i) = error  ;
end
%% plot test error rate 
plot(testErrorRate);
grid on ;
title('mnist CNN');
xlabel('epoch');
ylabel('test error rate');

%% plot mean squared error
% figure; plot(cnn.rL);

% assert(er<0.12, 'Too big error');
