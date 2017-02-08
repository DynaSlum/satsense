% simplest default training and testing of an Image category classifier
% using SURF detector and descriptor on Bag of Visual Words (BoVW)
% classifiation framework and multiclass SVM classifier 

%% setup parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

base_path = fullfile(root_dir, 'Data','Kalyan', 'Datasets3Classes');
sav_path = fullfile(root_dir, 'Results','Classification3Classes','DatastoresAndFeatures');

tile_sizes = [100];
tile_sizes_m = [80];
vocabulary_sizes = [10 20 50];
n = 1;
%num_datasets = length(tile_sizes);
tile_size = tile_sizes(n);
tile_size_m = tile_sizes_m(n);
str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
vocabulary_size = vocabulary_sizes(n);
%vocabulary_size = 500; % MATLAB default!

fractionTrain = 0.7;
fractionStrongestFeatures = 0.7;

point_selection = 'Detector';
surf_upright = false;

summary_flag = true;
preview_flag = true;
verbose = true;
visualize = false; % visualization still doesn't work!
%% create image datastore
    
disp(['Creating image data store for tile size: ' num2str(tile_size) ' pixels = ' num2str(tile_size_m) ' meters.']);

image_dataset_location = fullfile(base_path,str);
[imds] = createImageDatastore( image_dataset_location, summary_flag,...
    preview_flag);
disp('-----------------------------------------------------------------');


%% split image datastore
disp(['Splititng the datastore into Train and Test datastores with fractionTrain: ' num2str(fractionTrain*100) '%']);
[imdsTrain, imdsTest] = splitImageDatastore(imds,...
    fractionTrain, summary_flag);
disp('-----------------------------------------------------------------');

%% Create Bag of Visual words

 disp(['Creating features (BoVW) using vocabulary size ' num2str(vocabulary_size)...
     ' for the Train datastore keeping the '...
     num2str(fractionStrongestFeatures*100) '% strongest features']);
 bagVW = bagOfFeatures(imdsTrain, 'VocabularySize',vocabulary_size,...
    'StrongestFeatures', fractionStrongestFeatures, 'PointSelection',point_selection,...
    'Upright', surf_upright);
% [bagVWTrain, feature_vectors_train] = createVisualVocabulary( imdsTrain,...
%                 vocabulary_size, fractionStrongestFeatures, [], false, verbose, visualize);
% disp(['Creating features (BoVW) for the Test datastore.keeping the ' num2str(fractionStrongestFeatures*100) '% strongest features']);
% [bagVWTest, feature_vectors_test] = createVisualVocabulary( imdsTest,...
%                 vocabulary_size, fractionStrongestFeatures, [], false, verbose, visualize);
disp('-----------------------------------------------------------------');

%% Train a classifier
disp('Training an image categiry classifier');
categoryClassifier = trainImageCategoryClassifier(imdsTrain,bagVW);
disp('-----------------------------------------------------------------');

%% Evaluate Classifier's perfomance
disp('Evaluating perfomance on the Training set');
confMatrixTrain = evaluate(categoryClassifier, imdsTrain);
disp('Evaluating perfomance on the Testing set');
confMatrixTest = evaluate(categoryClassifier, imdsTest);

% Compute average accuracy
mean(diag(confMatrixTest));
