% Testing of slumTiling and nonSlumTiling

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

base_path = fullfile(root_dir, 'Data','Kalyan', 'Datasets3Classes');
factor = 0.8;
tile_sizes = 100;
tile_sizes_m = 80;
step_factor = 9;

num_datasets = length(tile_sizes);

data_path = fullfile(root_dir, 'Data','Kalyan','Rasterized_Lourens');
image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';
slum_mask = 'all_slums.tif';
builtup_mask = 'builtup_mask.tif';
nonbuiltup_mask = 'nonbuiltup_mask.tif';

%% create datasets
for n = 1: num_datasets
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    stepY = floor(tile_size/step_factor);
    stepX = stepY;
    tile_step = [stepX stepY];
    
    image_fullfname = fullfile(data_path, image_fname);
    mask_fullfname = fullfile(data_path, slum_mask);
    masks_fullfnames = {fullfile(data_path, builtup_mask),...
        fullfile(data_path, nonbuiltup_mask)};
    
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    tiles_path = fullfile(base_path, str);
    
    % tile the image
    disp(['Tiling for dataset ' num2str(n) '...']);
    [number_tiles]= slumTiling( image_fullfname, [tile_size tile_size], tile_step, factor, ...
    mask_fullfname, tiles_path);
    disp(['There are ', num2str(number_tiles), ' number of images for class Slum.']);
    
    [number_images] = nonSlumTiling( image_fullfname, number_tiles,...
    [tile_size tile_size], tile_step, factor, masks_fullfnames, tiles_path);
    
    disp(['There are ' , num2str(number_images.builtup), ' number of images for class BuiltUp.']);
    disp(['There are ' , num2str(number_images.nonbuiltup),' number of images for class NonBuiltUp.']);
    
    disp('Done!');
    disp('---------------------------------');
end


disp('DONE!');
