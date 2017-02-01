% Testing class_perf_test - computing performance of a clsassifier on a
% test set

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum','Results');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum','Results');
end

base_features_path = fullfile(root_dir, 'Classification3Classes','DatastoresAndFeatures');
base_classif_path = fullfile(root_dir, 'Classification3Classes', 'Classifiers');
% tile_sizes = [417 333 250 167];
% tile_sizes_m = [250 200 150 100];

tile_sizes = [417 333 250];
tile_sizes_m = [250 200 150];


num_datasets = length(tile_sizes);

%% get the respective classifiers and test on feature datasets
for n = 1 : num_datasets
    
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    dataset_features_location = fullfile(base_features_path,str, 'FeatureTableTest.mat');
    
    classifier_location = fullfile(base_classif_path, ['trainedClassifier_' str]);
    
    load(dataset_features_location); % contains feature_table
    load(classifier_location); % contains trainedClassifier
    
    disp(['Computing Best classifier performance for image testing data store # ', num2str(n), ' out of ', ...
            num2str(num_datasets)]);
    [~,~, perf_stats] = class_perf_test(feature_table, trainedClassifier);
    disp('Confusion matrix: ');
    disp(perf_stats.confusionMat);
end