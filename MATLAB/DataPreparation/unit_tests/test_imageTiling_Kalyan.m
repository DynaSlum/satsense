% Testing of imageTiling for Kalyan
%% parameters
[ paths, processing_params, exec_flags] = config_params_Kalyan();

[data_dir, masks_dir, ~, ~] = v2struct(paths);
[~, ~, ~, ~, ~, ~, roi] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

base_tiles_path = fullfile(data_dir, 'Datasets4MATLAB');
factor = 0.8;
save_mixed = false;
tile_sizes_m = [50 100 150 200 250];
tile_sizes = [84 167 250 334 417];

num_datasets = length(tile_sizes);

%% create datasets
slum_mask = strcat('Kalyan_', roi, '_slumMask.tif');
builtup_mask = strcat('Kalyan_',roi,'_urbanMask.tif');
nonbuiltup_mask = strcat('Kalyan_',roi,'_vegetationMask.tif');


image_fullfname = fullfile(data_dir, ['Mumbai_P4_R1C1_3_ROI_fixed_clipped.tif']);
masks_fullfnames = {fullfile(masks_dir, slum_mask), ...
    fullfile(masks_dir, builtup_mask),...
    fullfile(masks_dir, nonbuiltup_mask)};

for n = 1:num_datasets
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    stepY = floor(tile_size/2);
    stepX = stepY;
    tile_step = [stepX stepY];
    
    
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    tiles_path = fullfile(base_tiles_path, str, roi);
    
    if exist(tiles_path,'dir')==7
        rmdir(tiles_path,'s');
    end
    
    % tile the image
    disp(['Tiling for dataset ' num2str(n) ':' str '...']);
    [number_images]= imageTiling( image_fullfname, [tile_size tile_size], tile_step, factor, ...
        masks_fullfnames, tiles_path);
    disp(['There are ', num2str(number_images.slum), ' number of images for class Slum.']);
    disp(['There are ' , num2str(number_images.builtup), ' number of images for class BuiltUp.']);
    disp(['There are ' , num2str(number_images.nonbuiltup),' number of images for class NonBuiltUp.']);
    if save_mixed
        disp(['There are ' , num2str(number_images.mixed), ' number of images for class Mixed.']);
    else
        disp(['There are ' , num2str(number_images.mixed), ' number of images for class Mixed, but they are not saved!.']);
    end
    disp(['Done for dataset: '  num2str(n) '!']);
    disp('---------------------------------');
end

disp('DONE!');
