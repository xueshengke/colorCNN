function [ totalAccuracy ] = displayAccuracy(labels, bad)
    wrongNumber = size(bad, 2);
    classNumber = size(labels, 1);
    totalNumber = size(labels, 2);
    totalAccuracy = 1 - wrongNumber / totalNumber;
    fprintf('Total Accuracy : %.2f %% \n', 100 * totalAccuracy);
    
    wrongLabel = labels(:, bad);
    classWrong = sum(wrongLabel, 2);
    for i = 1 : classNumber
        fprintf('Accuracy of class %d : %.2f %% \n', i, ...
            100 * (1 - classWrong(i) ./ (totalNumber / classNumber)));
    end
               
end