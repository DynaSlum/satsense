% Testing of imageTiling for Bangalore

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
    data_root_dir = fullfile(root_dir, 'Data','Bangalore','GEImages');
end

data_dir = fullfile(data_root_dir, 'Clipped','fixed');
masks_dir = fullfile(data_root_dir, 'masks');

base_tiles_path = fullfile(data_root_dir, 'Datasets4MATLAB');
factor = 0.8;
save_mixed = false;
tile_sizes = [67 134 200 268 334 400];
%these are approx! real are 10.05 20.1 30 40.2, 50.1  and 60
tile_sizes_m = [10 20 30 40 50 60]; 

ROIs = {
    'ROI1'
    'ROI2'
    'ROI3'
    'ROI4'
    'ROI5'
};

num_datasets = length(tile_sizes);
num_ROIs = length(ROIs);


%% create datasets
for r = 1:num_ROIs
    roi = ROIs{r};
    disp(strcat('Creating tile sets for ROI: ', roi));
    base_path = fullfile(data_dir, roi);
    
    slum_mask = strcat('Bangalore_', roi, '_slumMask.tif');
    builtup_mask = strcat('Bangalore_',roi,'_urbanMask.tif');
    nonbuiltup_mask = strcat('Bangalore_',roi,'_vegetationMask.tif');
    
    
    image_fullfname = fullfile(data_dir, strcat('Bangalore_', roi, '.tif'));
    masks_fullfnames = {fullfile(masks_dir, slum_mask), ...
        fullfile(masks_dir, builtup_mask),...
        fullfile(masks_dir, nonbuiltup_mask)};
        
    for n = 1: num_datasets
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
    disp(['Done for ROI:' roi '!']);
    disp('+++++++++++++++++++++++++++++++++++++++');
end

disp('DONE!');
