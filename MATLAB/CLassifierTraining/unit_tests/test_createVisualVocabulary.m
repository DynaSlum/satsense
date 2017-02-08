% Testing createVisualVocabulary.m

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

%base_path = fullfile(root_dir, 'Results','Classification4ClassesInclMixed','DatastoresAndFeatures');
% tile_sizes = [417 333 250 167 83];
% tile_sizes_m = [250 200 150 100 50];

base_path = fullfile(root_dir, 'Results','Classification3Classes','DatastoresAndFeatures');
% tile_sizes = [417 333 250 167 ];
% tile_sizes_m = [250 200 150 100];

tile_sizes = [100 ];
tile_sizes_m = [80 ];

num_datasets = length(tile_sizes);
vocabulary_size = 50; %10; %20;

visualize = false; % visualization still doesn't work!
save_flag = true;
verbose = true;

train = true;
test = true;

%% create the bag of VW for each datastore
for n = 1 : num_datasets
    %vocabulary_size = ceil(50*num_datasets/n);
    
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    image_dataset_location = fullfile(base_path,str);
    datastore_file = fullfile(image_dataset_location, 'imds.mat');
    if train
        disp(['Creating Visual Vocabulary for image training data store # ', num2str(n), ' out of ', ...
            num2str(num_datasets)]);
        load(datastore_file, 'imdsTrain');
        
        [bagVW, feature_vectors] = createVisualVocabulary( imdsTrain,...
            vocabulary_size, 0.7, [], false, verbose, visualize);
        
        if save_flag
            sav_file = fullfile(image_dataset_location, ['Bo' num2str(vocabulary_size) 'VWTrain.mat']);
            save(sav_file, 'bagVW', 'feature_vectors');
        end
    end
    if test
        disp(['Creating Visual Vocabulary for image testing data store # ', num2str(n), ' out of ', ...
            num2str(num_datasets)]);
        
        load(datastore_file, 'imdsTest');
        
        [bagVW, feature_vectors] = createVisualVocabulary( imdsTest,...
            vocabulary_size, 0.7, [], false, verbose, visualize);
        if save_flag
            sav_file = fullfile(image_dataset_location, ['Bo' num2str(vocabulary_size) 'VWTest.mat']);
            save(sav_file, 'bagVW', 'feature_vectors');
        end
    end
    
    
    
    disp('-----------------------------------------------------------------');
end
disp('DONE.');