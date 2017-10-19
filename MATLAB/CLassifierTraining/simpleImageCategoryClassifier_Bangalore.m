% simplest default training and testing of an Image category classifier
% using SURF detector and descriptor on Bag of Visual Words (BoVW)
% classifiation framework and multiclass SVM classifier on the 2 ROIs from
% GE Images of Bangalore

%% setup parameters

if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
    data_root_dir = fullfile(root_dir, 'Data','Bangalore','GEImages');
    results_dir = fullfile(root_dir, 'Results','Bangalore', 'Classification3Classes');
    if not(exist(results_dir,'dir')==7)
        mkdir(results_dir);
    end
    
    sav_path_datastores = fullfile(results_dir, 'DatastoresAndFeatures');
    sav_path_classifier = fullfile(results_dir, 'Classifiers');
    
end

data_dir = fullfile(data_root_dir, 'Clipped','fixed');
masks_dir = fullfile(data_root_dir, 'masks');

base_tiles_path = fullfile(data_root_dir, 'Datasets4MATLAB');
factor = 0.8;
save_mixed = false;
tile_sizes = [67 134 200 268 334 400];
%these are approx! real are 10.05 20.1 30 40.2, 50.1  and 60
tile_sizes_m = [10 20 30 40 50 60];

ROIs = {
    'ROI1'
    'ROI2'
    'ROI3'
    'ROI4'
    'ROI5'
    };

num_datasets = length(tile_sizes);
num_ROIs = length(ROIs);


%% feature parameters

vocabulary_sizes = [10 20 50];

fractionTrain = 0.8;
fractionStrongestFeatures = 0.8;

point_selection = {'Detector'};

surf_upright = false;


%% execution flags
summary_flag = true;
preview_flag = false; % preview of the datastore
verbose = true;
visualize = false; % visualization still doesn't work!
sav = true;

if sav
    if not(exist(sav_path_datastores,'dir')==7)
        mkdir(sav_path_datastores);
    end
    
    if not(exist(sav_path_classifier,'dir')==7)
        mkdir(sav_path_classifier);
    end
end

%% create datastores, train and test classifier
for d = 1: num_datasets
    %% create image datastores
    tile_size = tile_sizes(d);
    tile_size_m = tile_sizes_m(d);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    
    disp(['Creating datastore for dataset ' num2str(d) ':' str '...']);
    
    for r = 1:num_ROIs
        roi = ROIs{r};
       % disp(strcat('Using tile sets for ROI: ', roi));
        
        tiles_path = fullfile(base_tiles_path, str, roi);
        image_dataset_location{r} = tiles_path;
    end
    
    [imds] = createImageDatastore( image_dataset_location, summary_flag, preview_flag);
    
    %% balance the dataset
   % disp('Balancing the datasets.');
    tbl = countEachLabel(imds);
    minSetCount = min(tbl{:,2});
    imds = splitEachLabel(imds, minSetCount, 'randomize');
    countEachLabel(imds)
    
    if sav
        fname = fullfile(sav_path_datastores, ['datastore_' str '.mat']) ;
        save(fname, 'imds');
    end
    
    disp('-----------------------------------------------------------------');
    
    
    
    %% split image datastore
    disp(['Splititng the datastore into Train and Test datastores with fractionTrain: ' num2str(fractionTrain*100) '%']);
    [imdsTrain, imdsTest] = splitImageDatastore(imds, fractionTrain, summary_flag);
    disp('-----------------------------------------------------------------');
    
    %% loop over parameters
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
            fname = fullfile(sav_path_classifier, ['trained_SURF_SVM_Classifier' str '_' num2str(vocabulary_size) '_' roi '_' str '.mat']) ;
            save(fname, 'categoryClassifier');
        end
        disp('Paused. Press any key to continue');
        pause;
    end % vocabulary sizes
    
end % for num_datasets
    
