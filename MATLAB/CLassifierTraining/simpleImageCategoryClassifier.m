% simplest default training and testing of an Image category classifier
% using SURF detector and descriptor on Bag of Visual Words (BoVW)
% classifiation framework and multiclass SVM classifier

%% setup parameters
if isunix
    root_dir = fullfile('/home','bweel','Documents','projects','dynaslum', 'SatelliteImaging','MATLAB');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

data_dir = '/home/bweel/Documents/projects/dynaslum/data/Bangalore/GE_Images/';
roi = 'ROI5';
base_path = fullfile(data_dir, roi);
sav_path_datastores = fullfile(data_dir, 'Results','Classification3Classes','DatastoresAndFeatures');
sav_path_classifier = fullfile(data_dir, 'Results','Classification3Classes','Classifiers');

tile_sizes = [135];
tile_sizes_m = [20];
%vocabulary_sizes = [10 20 50];
vocabulary_sizes = [50];
n = 1;
%num_datasets = length(tile_sizes);
tile_size = tile_sizes(n);
tile_size_m = tile_sizes_m(n);
str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
%vocabulary_size = vocabulary_sizes(n);

fractionTrain = 0.3;
fractionStrongestFeatures = 0.8;

%point_selections = {'Detector','Grid'};
point_selections = {'Detector'};

surf_upright = false;

summary_flag = true;
preview_flag = true; % preview of the datastore
verbose = true;
visualize = false; % visualization still doesn't work!
sav = true;
%% create image datastore

disp(roi);
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
disp(['Splititng the datastore into Train and Test datastores with fractionTrain: ' num2str(fractionTrain*100) '%']);
[imdsTrain, imdsTest] = splitImageDatastore(imds, fractionTrain, summary_flag);
disp('-----------------------------------------------------------------');

%% loop over parameters
for point_selection = point_selections
    for vocabulary_size = vocabulary_sizes
        %% Create Bag of Visual words
        
        disp(['Creating features (BoVW) using vocabulary size ' num2str(vocabulary_size)...
            ' and SUFR point locations ', char(point_selection) ...
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
        [confmatTrain] = evaluate(categoryClassifier, imdsTrain);
        perf_stats_train = confusionmatStats(confmatTrain);
        TrT = table( perf_stats_train.accuracy*100, perf_stats_train.sensitivity*100,...
            perf_stats_train.specificity*100, perf_stats_train.precision*100, ...
            perf_stats_train.recall*100, perf_stats_train.Fscore,...
            'RowNames', cellstr(unique(imdsTrain.Labels)),...
            'VariableNames', {'accuracy';'sensitivity'; 'specificity';...
            'precision';'recall';'Fscore'});
        disp(TrT);
        disp('-----------------------------------------------------------------');
        disp('Evaluating perfomance on the Test set');
        [confmatTest] = evaluate(categoryClassifier, imdsTest);
        perf_stats_test = confusionmatStats(confmatTest);
        TrTs = table( perf_stats_test.accuracy*100, perf_stats_test.sensitivity*100,...
            perf_stats_test.specificity*100, perf_stats_test.precision*100, ...
            perf_stats_test.recall*100, perf_stats_test.Fscore,...
            'RowNames', cellstr(unique(imdsTest.Labels)),...
            'VariableNames', {'accuracy';'sensitivity'; 'specificity';...
            'precision';'recall';'Fscore'});
        disp(TrTs);
        disp('-----------------------------------------------------------------');
        % save the trained classifier
        if sav
            fname = fullfile(sav_path_classifier, ['trained_SURF_SVM_Classifier' num2str(vocabulary_size) '_' roi '_' str '.mat']) ;
            save(fname, 'categoryClassifier');
        end
    end
end

