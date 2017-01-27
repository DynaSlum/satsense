% test_plot_accuracy_models

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

base_path = fullfile(root_dir, 'Results','Classification3Classes','PerformanceComparision');
tile_sizes = [417 333 250 167];
tile_sizes_m = [250 200 150 100];

num_datasets = length(tile_sizes);

%% display accuracy plots for each dataset
for n = 1 : num_datasets
     disp(['Creating the best classifier Precission and Recall plot image training data store # ', num2str(n), ' out of ', ...
            num2str(num_datasets)])
    % load the data table containing the performance measures
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    res_str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    str = [num2str(n) res_str];
    fname = ['ClassifPerfDataset' str 'TrainValTable.mat'];
    perf_dataset_file = fullfile(base_path,fname); 
    
    load(perf_dataset_file);
    varname = ['ClassifPerfDataset' str 'TrainVal'];
    data_table = eval(varname);   
    
    % plot the accuracy of the best model for all classes    
    
    switch n
        case 1
            best_model_ind = 21;
        case {2, 3, 4}
            best_model_ind = 12;
    end
    title_str = ['Validation performance best "Slum" classifier for ' res_str ' tiles'];
            
    [fig_h] = plot_precission_recall_best_model_3classes( data_table, best_model_ind, ...
        [0 1 0],[],title_str);
    
end


