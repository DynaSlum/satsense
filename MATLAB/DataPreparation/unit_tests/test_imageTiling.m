% Testing of imageTiling

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

% base_path = fullfile(root_dir, 'Data', 'Kalyan', 'Datasets4ClassesInclMixed');
% factor = 0.52;
% save_mixed = true;
% tile_sizes = [417 333 250 167 83];
% tile_sizes_m = [250 200 150 100 50];

base_path = fullfile(root_dir, 'Data','Kalyan', 'Datasets3Classes');
factor = 0.75;
save_mixed = false;
tile_sizes = [417 333 250 167];
tile_sizes_m = [250 200 150 100];


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
    disp('Done!');
    disp('---------------------------------');
end


disp('DONE!');
