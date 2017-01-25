% Testing createImageDatastore.m

%% parameters

if isunix
    root_dir = fullfile('home','elena','DynaSlum');
else
    root_dir = fullfile('C','Projects', 'DynaSlum');
end

% base_path = 'C:\Projects\DynaSlum\Data\Kalyan\Datasets4ClassesInclMixed\';
% sav_path = 'C:\Projects\DynaSlum\Results\Classification4ClassesInclMixed\DatastoresAndFeatures\';
% tile_sizes = [417 333 250 167 83];
% tile_sizes_m = [250 200 150 100 50];

base_path = fullfile(root_dir, 'Results','Classification3Classes','DatastoresAndFeatures');
sav_path = fullfile(root_dir, 'Data','Kalyan', 'Datasets3Classes', 'DatastoresAndFeatures');
tile_sizes = [417 333 250 167];
tile_sizes_m = [250 200 150 100];

num_datasets = length(tile_sizes);

summary_flag = true;
preview_flag = false;

save_flag = true;

%% create image datastore and show summary and sample of the 4 classes
for n = 1: num_datasets
    disp(['Creating image data store # ', num2str(n), ' out of ', ...
        num2str(num_datasets)]);
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    image_dataset_location = fullfile(base_path,str);
    [imds] = createImageDatastore( image_dataset_location, summary_flag,...
        preview_flag);
    if save_flag
        image_datastore_location = fullfile(sav_path,str);
        sav_file = fullfile(image_datastore_location, 'imds.mat');
        save(sav_file, 'imds');
    end
    disp('-----------------------------------------------------------------');
end
disp('DONE.');