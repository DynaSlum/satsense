% Testing of imageTiling

%% parameters
tile_size = [333 333];
factor = 0.52;
data_path = 'C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\';
image_fname = 'Mumbai_P4_R1C1_3_clipped.tif';
slum_mask = 'slums_municipality_raster_mask_8.tif';
urban_mask = 'urban_mask.tif';
rural_mask = 'rural_mask.tif';

image_fullfname = fullfile(data_path, image_fname);
masks_fullfnames = {fullfile(data_path, slum_mask), ...
    fullfile(data_path, urban_mask),...
    fullfile(data_path, rural_mask)};

tiles_path = 'C:\Projects\DynaSlum\Data\Kalyan\Datasets\px333m200';

%% tile the image
disp('Tiling...');
imageTiling( image_fullfname, tile_size, factor, ...
    masks_fullfnames, tiles_path);
disp('Done!');
    