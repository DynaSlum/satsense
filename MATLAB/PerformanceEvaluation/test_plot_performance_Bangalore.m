% test_plot_performance_Bangalore

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
    
    %sav_path_datastores = fullfile(results_dir, 'DatastoresAndFeatures');
    %sav_path_classifier = fullfile(results_dir, 'Classifiers');
    sav_path_performance = fullfile(results_dir, 'Performance');
    
end

tile_sizes = [67 134 200 268 334 400];
%these are approx! real are 10.05 20.1 30 40.2, 50.1  and 60
tile_sizes_m = [10 20 30 40 50 60];

num_datasets = length(tile_sizes);

%% feature parameters

vocabulary_sizes = [10 20 50];

%% execution flags
verbose = false;
sav = false;

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
                
        fname = fullfile(sav_path_performance, ['performance_train_' num2str(vocabulary_size) '_' str '.mat']) ;
        load(fname, 'TrT');
        accuracy_train(v,d,:) = TrT.accuracy;
        sensitivity_train(v,d,:) = TrT.sensitivity;
        specificity_train(v,d,:) = TrT.specificity;
        precision_train(v,d,:) = TrT.precision;
        fscore_train(v,d,:) = TrT.Fscore;
        
        fname = fullfile(sav_path_performance, ['performance_test_' num2str(vocabulary_size) '_' str '.mat']) ;
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
    sensitivity_train, specificity_train, precision_train);

plot_acc_sens_spec_prec_all( accuracy_test, ...
    sensitivity_test, specificity_test, precision_test); 

plot_fscore_all( fscore_train);
plot_fscore_all( fscore_test);