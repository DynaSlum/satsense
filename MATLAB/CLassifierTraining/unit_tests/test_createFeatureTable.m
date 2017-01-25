% Testing createFeatureTable.m

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

% base_path = fullfile(root_dir, 'Results','Classification4ClassesInclMixed','DatastoresAndFeatures');
% tile_sizes = [417 333 250 167 83];
% tile_sizes_m = [250 200 150 100 50];

base_path = fullfile(root_dir, 'Results','Classification3Classes','DatastoresAndFeatures');
tile_sizes = [417 333 250 167];
tile_sizes_m = [250 200 150 100];


num_datasets = length(tile_sizes);

save_flag = true;
verbose = true;

train = true;
test = true;
%% create the table of features for each datastore
for n = 1 : num_datasets
    
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    image_dataset_location = fullfile(base_path,str);
    datastore_file = fullfile(image_dataset_location, 'imds.mat');
    
    if train
        disp(['Creating Feature Table for image training data store # ', num2str(n), ' out of ', ...
            num2str(num_datasets)]);
        load(datastore_file, 'imdsTrain');
        
        % load the feature vectors
        features_file = fullfile(image_dataset_location, 'BoVWTrain.mat');
        load(features_file, 'feature_vectors');
        
        % create table
        [feature_table] = createFeatureTable( imdsTrain, feature_vectors);
        
        %% save
        if save_flag
            sav_file = fullfile(image_dataset_location, 'FeatureTableTrain.mat');
            save(sav_file, 'feature_table');
        end
    end
    if test
        disp(['Creating Feature Table for image testing data store # ', num2str(n), ' out of ', ...
            num2str(num_datasets)]);
        load(datastore_file, 'imdsTest');
        
        % load the feature vectors
        features_file = fullfile(image_dataset_location, 'BoVWTest.mat');
        load(features_file, 'feature_vectors');
        
        % create table
        [feature_table] = createFeatureTable( imdsTest, feature_vectors);
        
        %% save
        if save_flag
            sav_file = fullfile(image_dataset_location, 'FeatureTableTest.mat');
            save(sav_file, 'feature_table');
        end
    end    
    
    disp('-----------------------------------------------------------------');
end
disp('DONE.');