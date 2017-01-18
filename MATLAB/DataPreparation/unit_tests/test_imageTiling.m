% Testing of imageTiling

%% parameters
base_path = 'C:\Projects\DynaSlum\Data\Kalyan\Datasets4ClassesInclMixed\';
factor = 0.52;

tile_sizes = [417 333 250 167 83];
tile_sizes_m = [250 200 150 100 50];

num_datasets = length(tile_sizes);

data_path = 'C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\';
image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';
slum_mask = 'all_slums.tif';
builtup_mask = 'builtup_mask.tif';
nonbuiltup_mask = 'nonbuiltup_mask.tif';

%% create datasets
for n = 1: num_datasets
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    stepY = floor(tile_size/4);
    stepX = stepY;
    tile_step = [stepX stepY];
    
    image_fullfname = fullfile(data_path, image_fname);
    masks_fullfnames = {fullfile(data_path, slum_mask), ...
        fullfile(data_path, builtup_mask),...
        fullfile(data_path, nonbuiltup_mask)};
    
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    tiles_path = fullfile(base_path, str);
    
    % tile the image
    disp(['Tiling for dataset ' num2str(n) '...']);
    imageTiling( image_fullfname, [tile_size tile_size], tile_step, factor, ...
        masks_fullfnames, tiles_path);
    disp('Done!');
    disp('---------------------------------');
end

disp('DONE!');
