% Testing of imageTiling

%% parameters
tile_size = [333 333];
stepY = floor(333/2);
stepX = floor(stepY/2);
tile_step = [stepX stepY];
factor = 0.52;
data_path = 'C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\';
image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';
slum_mask = 'all_slums.tif';
builtup_mask = 'builtup_mask.tif';
nonbuiltup_mask = 'nonbuiltup_mask.tif';

image_fullfname = fullfile(data_path, image_fname);
masks_fullfnames = {fullfile(data_path, slum_mask), ...
    fullfile(data_path, builtup_mask),...
    fullfile(data_path, nonbuiltup_mask)};

tiles_path = 'C:\Projects\DynaSlum\Data\Kalyan\Datasets\px333m200';

%% tile the image
disp('Tiling...');
imageTiling( image_fullfname, tile_size, tile_step, factor, ...
    masks_fullfnames, tiles_path);
disp('Done!');
    