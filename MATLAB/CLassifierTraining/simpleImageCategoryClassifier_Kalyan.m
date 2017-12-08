% simplest default training and testing of an Image category classifier
% using SURF detector and descriptor on Bag of Visual Words (BoVW)
% classifiation framework and multiclass SVM classifier on the ROI from
% the RGB image of Kalyan

%% parameters
[ paths, processing_params, exec_flags] = config_params_Kalyan();

[data_dir, masks_dir, datastores_dir, classifier_dir, performance_dir,~] = v2struct(paths);
[~, ~, ~, ~, ~, ~, roi] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

base_tiles_path = fullfile(data_dir, 'Datasets4MATLAB');
tile_sizes_m = [50 100 150 200];
tile_sizes = [84 167 250 334];
    
num_datasets = length(tile_sizes);


%% feature parameters

vocabulary_sizes = [10 20 50];

fractionTrain = 0.8;
fractionStrongestFeatures = 0.8;

point_selection = {'Detector'};

surf_upright = false;


%% execution flags
summary_flag = true;
preview_flag = false; % preview of the datastore

if sav
    if not(exist(datastores_dir,'dir')==7)
        mkdir(datastores_dir);
    end
    
    if not(exist(classifier_dir,'dir')==7)
        mkdir(classifier_dir);
    end
    
    if not(exist(performance_dir,'dir')==7)
        mkdir(performance_dir);
    end
end

%% create datastores, train and test classifier
for d = 1: num_datasets
    %% create image datastores
    tile_size = tile_sizes(d);
    tile_size_m = tile_sizes_m(d);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    
    disp(['Creating datastore for dataset ' num2str(d) ':' str '...']);
    
     
    tiles_path = fullfile(base_tiles_path, str, roi);
    image_dataset_location = tiles_path;
    
    [imds] = createImageDatastore( image_dataset_location, summary_flag, preview_flag);
    
    %% balance the dataset
   % disp('Balancing the datasets.');
    tbl = countEachLabel(imds);
    minSetCount = min(tbl{:,2});
    imds = splitEachLabel(imds, minSetCount, 'randomize');
    countEachLabel(imds)
    
    if sav
        fname = fullfile(datastores_dir, ['datastore_' str '.mat']) ;
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
        
        % save the performance results
        if sav
            disp('Evaluating perfomance on the Training set');
            fname = fullfile(performance_dir, ['performance_train_' num2str(vocabulary_size) '_' str '.mat']) ;
            save(fname, 'confmatTrain', 'TrT');
        end
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
        % save the performance results
        if sav
            disp('Saving perfomance on the Test set');
            fname = fullfile(performance_dir, ['performance_test_' num2str(vocabulary_size) '_' str '.mat']) ;
            save(fname, 'confmatTest', 'TrTs');
        end
        disp('-----------------------------------------------------------------');
        % save the trained classifier
        if sav
            fname = fullfile(classifier_dir, ['trained_SURF_SVM_Classifier_' num2str(vocabulary_size) '_' str '.mat']) ;
            save(fname, 'categoryClassifier');
        end

    end % vocabulary sizes
    disp('**********************************************************************************')
    disp('Paused. Press any key to continue');
    %pause;   
end % for num_datasets
    
