% test_plot_performance_Kalyan

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


%% loading of saved performance results and arranging them in data structures for display
v = 0;
for vocabulary_size = vocabulary_sizes
    if verbose
        disp(['Loading performance files for vocabulary size: ' num2str(vocabulary_size) '...']);
    end
    v = v+1;
    for d = 1: num_datasets
        %% create image datastores
        tile_size = tile_sizes(d);
        tile_size_m = tile_sizes_m(d);
        str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
        
        if verbose
            disp(['Loading performance files for dataset: ' str '...']);
        end
                
        fname = fullfile(performance_dir, ['performance_train_' num2str(vocabulary_size) '_' str '.mat']) ;
        load(fname, 'TrT');
        accuracy_train(v,d,:) = TrT.accuracy;
        sensitivity_train(v,d,:) = TrT.sensitivity;
        specificity_train(v,d,:) = TrT.specificity;
        precision_train(v,d,:) = TrT.precision;
        fscore_train(v,d,:) = TrT.Fscore;
        
        fname = fullfile(performance_dir, ['performance_test_' num2str(vocabulary_size) '_' str '.mat']) ;
        load(fname, 'TrTs');
        accuracy_test(v,d,:) = TrTs.accuracy;
        sensitivity_test(v,d,:) = TrTs.sensitivity;
        specificity_test(v,d,:) = TrTs.specificity;
        precision_test(v,d,:) = TrTs.precision;
        fscore_test(v,d,:) = TrTs.Fscore;
        
    end
end
        
%% generating performance figures
plot_acc_sens_spec_prec_all( accuracy_train, ...
    sensitivity_train, specificity_train, precision_train, tile_sizes_m);

plot_acc_sens_spec_prec_all( accuracy_test, ...
    sensitivity_test, specificity_test, precision_test, tile_sizes_m); 

plot_fscore_all( fscore_train, tile_sizes_m);
plot_fscore_all( fscore_test, tile_sizes_m);