% Testing splitImageDatastore.m

%% parameters
% base_path = 'C:\Projects\DynaSlum\Results\Classification4ClassesInclMixed\DatastoresAndFeatures\';
% tile_sizes = [417 333 250 167 83];
% tile_sizes_m = [250 200 150 100 50];

base_path = 'C:\Projects\DynaSlum\Results\Classification3Classes\DatastoresAndFeatures\';
tile_sizes = [417 333 250 167];
tile_sizes_m = [250 200 150 100];

fractionTrain = 0.7;
%fractionTest = 0.15;
num_datasets = length(tile_sizes);

summary_flag = true;
save_flag = true;

%% split the image datastore and show summary of each subset
for n = 1: num_datasets
    disp(['Splitting image data store # ', num2str(n), ' out of ', ...
        num2str(num_datasets)]);
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    image_dataset_location = fullfile(base_path,str);
    sav_file = fullfile(image_dataset_location, 'imds.mat');
    load(sav_file);    
    
    [imdsTrain, imdsTest] = splitImageDatastore(imds,...
    fractionTrain, summary_flag);

    if save_flag  
        save(sav_file, 'imdsTrain', 'imdsTest', '-append');
    end
    disp('-----------------------------------------------------------------');
end
disp('DONE.');