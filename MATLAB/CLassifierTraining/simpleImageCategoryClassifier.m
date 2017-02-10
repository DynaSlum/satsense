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
%vocabulary_size = vocabulary_sizes(n);

fractionTrain = 0.3;
fractionStrongestFeatures = 0.8;

point_selections = {'Detector','Grid'};

surf_upright = false;

summary_flag = true;
preview_flag = false; % preview of the datastore
verbose = true;
visualize = false; % visualization still doesn't work!
%% create image datastore
    
disp(['Creating image data store for tile size: ' num2str(tile_size) ' pixels = ' num2str(tile_size_m) ' meters.']);

image_dataset_location = fullfile(base_path,str);
[imds] = createImageDatastore( image_dataset_location, summary_flag, preview_flag);
disp('-----------------------------------------------------------------');

%% balance the dataset
disp('Balancing the datasets.');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)

%% split image datastore
disp(['Splititng the datastore into Train and Validation datastores with fractionTrain: ' num2str(fractionTrain*100) '%']);
[imdsTrain, imdsVal] = splitImageDatastore(imds, fractionTrain, summary_flag);
disp('-----------------------------------------------------------------');


for point_selection = point_selections
    for vocabulary_size = vocabulary_sizes
        %% Create Bag of Visual words
        
        disp(['Creating features (BoVW) using vocabulary size ' num2str(vocabulary_size)...
            'and SUFR point locations ', char(point_selection) ...
            ' for the Train datastore keeping the ' ...
            num2str(fractionStrongestFeatures*100) '% strongest features']);
        bagVW = bagOfFeatures(imdsTrain, 'VocabularySize',vocabulary_size, ...
            'StrongestFeatures', fractionStrongestFeatures, ...
            'PointSelection', char(point_selection), ...
            'Upright', surf_upright);
        
        disp('-----------------------------------------------------------------');
        
        %% Train a classifier
        disp('Training an image categiry classifier');
        categoryClassifier = trainImageCategoryClassifier(imdsTrain,bagVW);
        disp('-----------------------------------------------------------------');
        
        %% Evaluate Classifier's perfomance
        disp('Evaluating perfomance on the Training set');
        confMatrixTrain = evaluate(categoryClassifier, imdsTrain);
        disp('Evaluating perfomance on the Validation set');
        confMatrixTest = evaluate(categoryClassifier, imdsVal);
        
        % Compute average accuracy
        mean(diag(confMatrixTest));
    end
end