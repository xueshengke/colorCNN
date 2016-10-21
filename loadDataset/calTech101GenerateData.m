function [dataTrain, labelsTrain, dataTest, labelsTest] = calTech101GenerateData()
%% data path
imagePath = '../dataset/Caltech101/101_ObjectCategories/';
className = {
    'Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'airplanes', ...
    'bonsai', 'car_side', 'chandelier', 'hawksbill', 'ketch'
};
imageSize = 200;
imageMap = 3;
classNumber = numel(className);
imageNumber = zeros(classNumber, 1);
imageSet = cell(classNumber, 1);
for i = 1 : classNumber
    fileList = dir(fullfile([imagePath, className{i}]));
    imageNumber(i) = length(fileList) - 2 ;
    disp([className{i}, ' has ', num2str(imageNumber(i)), ' images']);
    for j = 1 : imageNumber(i)
        img = imread([imagePath, className{i}, '/', fileList(j + 2).name]);
%        assert(length(size(img)) == 3, ['find image with dimension ' num2str(size(img))]);
         if length(size(img)) == 2
%             consImg = zeros(size(img, 1), size(img, 2), imageMap);
%             consImg(:, :, 1) = img;
%             consImg(:, :, 2) = img;
%             consImg(:, :, 3) = img;
%             img = consImg;
            img = repmat(img, [1 1 imageMap]);
         end
        img = double(img) / 255;
        img = imresize(img, [imageSize imageSize]);
        imageSet{i}.image(:, :, :, j) = img; 
    end
end

trainNumber = 50;
testNumber = 50;
dataTrain = zeros(imageSize, imageSize, imageMap, trainNumber * classNumber);
dataTest  = zeros(imageSize, imageSize, imageMap, testNumber  * classNumber);
labelsTrain = zeros(classNumber, trainNumber * classNumber);
labelsTest  = zeros(classNumber, testNumber  * classNumber);
labels = eye(classNumber);
for i = 1 : classNumber
    randNumber = randperm(imageNumber(i));
    imageSet{i}.image = imageSet{i}.image(:, :, :, randNumber);
    dataTrain(:, :, :, 1 + (i - 1) * trainNumber : i * trainNumber) = ...
        imageSet{i}.image(:, :, :, 1 : trainNumber);
    dataTest(:, :, :, 1 + (i - 1) * testNumber : i * testNumber) = ...
        imageSet{i}.image(:, :, :, trainNumber + 1 : trainNumber + testNumber);
    labelsTrain(:, 1 + (i - 1) * trainNumber : i * trainNumber) = ...
        repmat(labels(:, i), [1 trainNumber]);
    labelsTest(:, 1 + (i - 1) * testNumber : i * testNumber) = ...
        repmat(labels(:, i), [1 testNumber]);
end

% dataTrain = reshape(dataTrain, imageSize, imageSize, imageMap * trainNumber * classNumber);
% dataTest  = reshape(dataTest,  imageSize, imageSize, imageMap * testNumber  * classNumber);
% labelsTrain = reshape(labelsTrain, classNumber, imageMap * trainNumber * classNumber);
% labelsTest  = reshape(labelsTest,  classNumber, imageMap * testNumber  * classNumber);

randNumber = randperm(size(dataTrain, 4));
dataTrain = dataTrain(:, :, :, randNumber);
labelsTrain = labelsTrain(:, randNumber);

end