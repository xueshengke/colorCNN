function [dataTrain, labelsTrain, dataTest, labelsTest, dataUnlabeled] = stl10GenerateData()
%% load data in path
addpath('../dataset/stl10_matlab');

imageSize = 96;
imageMap = 3;
trainNumber = 5000;
testNumber = 8000;
unlabeledNumber = 100000;
classNumber = 10;
dataTrain = zeros(imageSize, imageSize, imageMap, trainNumber);
labelsTrain = zeros(classNumber, trainNumber);
dataTest = zeros(imageSize, imageSize, imageMap, testNumber);
labelsTest = zeros(classNumber, testNumber);

%% load train data
load train;
for i = 1 : numel(class_names)
    fprintf('Class %d :     %s \n', i, class_names{i});
end
X = double(X) ./ 255;
%% reconstruct labels
labelsTrain = full(sparse(y, 1 : trainNumber, 1));
%% reconstruct data
dataTrain = reshape(X', [imageSize imageSize imageMap trainNumber]);

%% load test data
load test;
X = double(X) ./ 255;
%% reconstruct labels
labelsTest = full(sparse(y, 1 : testNumber, 1));
%% reconstruct data
dataTest = reshape(X', [imageSize imageSize imageMap testNumber]);

%% unlabeled data requires large memory for 100000 images
load unlabeled;
X = double(X) ./ 255;
dataUnlabeled = reshape(X', [imageSize imageSize imageMap unlabeledNumber]);
clear X;
end