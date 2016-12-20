% Testing createImageDatastore.m

%% parameters
base_path = 'C:\Projects\DynaSlum\Data\Kalyan\Datasets\';
image_dataset_location = fullfile(base_path,'px417m250');
summary_flag = true;
preview_flag = true;


%% create image datastore and show summary and sample of the 4 classes
[imds] = createImageDatastore( image_dataset_location, summary_flag,...
    preview_flag);