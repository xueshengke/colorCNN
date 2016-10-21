function [dataTrain, labelsTrain, dataTest, labelsTest] = cifar100GenerateData()
%% load data in path
addpath('../dataset/cifar-100-matlab');

imageSize = 32;
imageMap = 3;
trainNumber = 50000;
testNumber = 10000;
classNumber = 100;
dataTrain = zeros(imageSize, imageSize, imageMap, trainNumber);
labelsTrain = zeros(classNumber, trainNumber);
dataTest = zeros(imageSize, imageSize, imageMap, testNumber);
labelsTest = zeros(classNumber, testNumber);

%% load train data
load train;
data = double(data) ./ 255; fine_labels = double(fine_labels);
%% reconstruct labels
fine_labels = fine_labels + 1;
labelsTrain = full(sparse(fine_labels, 1 : trainNumber, 1));
%% reconstruct data
dataTrain = reshape(data', [imageSize imageSize imageMap trainNumber]);

%% load test data
load test;
data = double(data) ./ 255; fine_labels = double(fine_labels);
%% reconstruct labels
fine_labels = fine_labels + 1;
labelsTest = full(sparse(fine_labels, 1 : testNumber, 1));
%% reconstruct data
dataTest = reshape(data', [imageSize imageSize imageMap testNumber]);

end