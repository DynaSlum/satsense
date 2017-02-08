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

% tile_sizes = [417 333 250];
% tile_sizes_m = [250 200 150];
vocabulary_sizes = [10 20 50];
tile_sizes = [100 ];
tile_sizes_m = [80 ];

num_datasets = length(tile_sizes);

%% get the respective classifiers and test on feature datasets
for n = 1 : num_datasets
    
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    
    for vocabulary_size = vocabulary_sizes
        disp(['Vocabulary size: ' num2str(vocabulary_size)]);
        
        dataset_features_location = fullfile(base_features_path,str, ['Feature' num2str(vocabulary_size) 'TableTest.mat']);
        
        classifier_location = fullfile(base_classif_path, ['trainedClassifier' num2str(vocabulary_size) '_' str]);
        
        load(dataset_features_location); % contains feature_table
        load(classifier_location); % contains trainedClassifier
        
        disp(['Computing Best classifier performance for image testing data store # ', num2str(n), ' out of ', ...
            num2str(num_datasets)]);
        [~,~, perf_stats] = class_perf_test(feature_table, trainedClassifier);
        disp('Classes: ');
        disp(perf_stats.groupOrder);
        disp('Confusion matrix: ');
        disp(perf_stats.confusionMat);
        disp('Accuracy, [%]: ');
        disp(perf_stats.accuracy*100);
        disp('Recall/TRP/Sensitivity, [%]: ');
        disp(perf_stats.sensitivity*100);
        disp('Precision/PPV, [%]: ');
        disp(perf_stats.precision*100);
        disp('----------------------------------------------------------------------------------');
    end
end