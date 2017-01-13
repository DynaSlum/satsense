function [imdsTrain, imdsTest] = splitImageDatastore( full_imds,...
    fractionTrain, summary_flag)


%% splitImageDatastore  wrapper around splitEachLabel for datasets with 
%   some preset parameters
%   INPUT:
%   full_imds -the datastore to be split in Train, Test and Validation sub-sets
%   fractionTrain -fraction (between [0 and 1)) for the training set
%   summary_flag- if true show thr label distribution per sets

%   OUPUT:
%   imdsTrain/Test - the split datastores
% For Testing use test_splitImageDatastore
% Note: see also https://nl.mathworks.com/help/vision/ug/...
% image-classification-with-bag-of-visual-words.html
%% split the full datastore
[imdsTrain, imdsTest] = splitEachLabel(full_imds, fractionTrain,'randomize');

%% Display Class Names and Counts
if summary_flag  
    tblTrain = countEachLabel(imdsTrain);                     %#ok
    disp('Training datastore: ');
    disp(tblTrain);
    tblTest = countEachLabel(imdsTest);                     %#ok
    disp('Testing datastore: ');
    disp(tblTest);
%     tblValidation = countEachLabel(imdsValidation);                %#ok
%     disp('Validaiton datastore: ');
%     disp(tblValidation);    
end

