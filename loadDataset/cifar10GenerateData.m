function [dataTrain, labelsTrain, dataTest, labelsTest] = cifar10GenerateData()
%% load data in path
addpath('../dataset/cifar-10-batches-mat');
load ('batches.meta.mat');
for i = 1 : numel(label_names)
    fprintf('Class %d :	%s \n', i, label_names{i});
end

batch = 5;
imageSize = 32;
imageMap = 3;
batchNumber = 10000;
trainNumber = 50000;
classNumber = 10;
dataTrain = zeros(imageSize, imageSize, imageMap, trainNumber);
labelsTrain = zeros(classNumber, trainNumber);
%% load 5 batches train data sequently
for i = 1 : batch
    load(['data_batch_' num2str(i) '.mat']);
    data = double(data) ./ 255; labels = double(labels);
    %% reconstruct labels
    labels = labels + 1;
    labelsTrainBatch = full(sparse(labels, 1 : batchNumber, 1));
    labelsTrain(:, 1 + (i - 1) * batchNumber : i * batchNumber) = labelsTrainBatch;
    %% reconstruct data
    dataTrainBatch = reshape(data', [imageSize imageSize imageMap batchNumber]);
    dataTrain(:, :, :, 1 + (i - 1) * batchNumber : i * batchNumber) = dataTrainBatch;
end
%% load test data
load test_batch ;
data = double(data) ./ 255; labels = double(labels);
dataTest = zeros(imageSize, imageSize, imageMap, batchNumber);
labelsTest = zeros(classNumber, batchNumber);
%% reconstruct labels
labels = labels + 1;
labelsTest = full(sparse(labels, 1 : batchNumber, 1));
%% reconstruct data
dataTest = reshape(data', [imageSize imageSize imageMap batchNumber]);
end
