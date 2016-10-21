function [dataTrain, labelsTrain, dataTest, labelsTest] = stlSubsetGenerateData()
%% load data in path
addpath('../dataset/stlSubset');
load stlTrainSubset;
load stlTestSubset;

%% reconstruct labels
labelsTrain = full(sparse(trainLabels, 1 : numTrainImages, 1));
labelsTest = full(sparse(testLabels, 1 : numTestImages, 1));

%% shuffle train data and labels    
% randNum = randperm(numTrainImages);
% dataTrain = trainImages(:, :, :, randNum);
% labelsTrain = labelsTrain(:, randNum);

dataTrain = trainImages;
dataTest = testImages;

% %% get parameters of images and labels
% imageSize = size(trainImages, 1);
% imageMaps = size(trainImages, 3);
% trainNumber = size(trainImages, 4);
% testNumber = size(testImages, 4);
% classNumber = max(trainLabels);
% %% reconstruct labels
% labelsTrain = zeros(classNumber, imageMaps, trainNumber);
% labels = eye(classNumber);
%  for i = 1 : trainNumber
%      labelsTrain(:, 1 : imageMaps, i) = repmat(labels(:, trainLabels(i)), [1 imageMaps]);
%  end
% 
% labelsTest = zeros(classNumber, imageMaps, testNumber);
% for i = 1 : testNumber
%     labelsTest(:, 1 : imageMaps, i) = repmat(labels(:, testLabels(i)), [1 imageMaps]);
% end
% %% reshape data and labels
% dataTrain = reshape(trainImages, imageSize, imageSize, imageMaps * trainNumber);
% dataTest  = reshape(testImages,  imageSize, imageSize, imageMaps * testNumber );
% labelsTrain = reshape(labelsTrain, classNumber, imageMaps * trainNumber);
% labelsTest  = reshape(labelsTest,  classNumber, imageMaps * testNumber );
% %% shuffle train data and labels    
% randNum = randperm(size(dataTrain, 3));
% dataTrain = dataTrain(:, :, randNum);
% labelsTrain = labelsTrain(:, randNum);
% 
% %% do not shuffle test data and labels
% % randNum = randperm(size(dataTest, 3));
% % dataTest = dataTest(:, :, randNum);
% % labelsTest = labelsTest(:, randNum);

end