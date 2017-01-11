function [imdsTrain, imdsTest, imdsValidation] = splitImageDatastore( full_imds,...
    fractionTrain, fractionTest, summary_flag)


%% splitImageDatastore  wrapper around splitEachLabel for datasets with 
%   some preset parameters
%   INPUT:
%   full_imds -the datastore to be split in Train, Test and Validation sub-sets
%   fractionTrain -fractio (between [0 and 1)) for the training set
%   fractionTest -fraction (between [0 and 1)) for the testing set
%   Note: what remains is validation, i.e. 
%   fractionValidation = 1- (fractionTrain +fractionTest)
%   summary_flag- if true show thr label distribution per sets

%   OUPUT:
%   imdsTrain/Test/Validation - the split datastores
% For Testing use test_splitImageDatastore
% Note: see also https://nl.mathworks.com/help/vision/ug/...
% image-classification-with-bag-of-visual-words.html
%% split the full datastore
[imdsTrain, imdsTest, imdsValidation] = splitEachLabel(full_imds, fractionTrain,...
    fractionTest,'randomize');

%% Display Class Names and Counts
if summary_flag  
    tblTrain = countEachLabel(imdsTrain);                     %#ok
    disp('Training datastore: ');
    disp(tblTrain);
    tblTest = countEachLabel(imdsTest);                     %#ok
    disp('Testing datastore: ');
    disp(tblTest);
    tblValidation = countEachLabel(imdsValidation);                %#ok
    disp('Validaiton datastore: ');
    disp(tblValidation);    
end

