%% setup parameters
berend = true;

if berend
    data_dir = '/home/bweel/Documents/projects/dynaslum/data/Bangalore/GE_Images/';
    masks_dir = fullfile(data_dir,'masks');
end

ROIS = {
    'ROI1'
    'ROI2'
    'ROI3'
    'ROI4'
    'ROI5'
};

% Tile Size in Pixels
tile_sizes = [135, 135];
% Corresponding tile size in meters
tile_sizes_m = [20];
% Step to take between tile
tile_step = [68, 68];
% Factor of true label to be present in the tile
factor = 0.9;
    
for i = 1:size(ROIS)
    disp('-----------------------------------------------------------');
    roi = ROIS{i};
    disp(strcat('Creating tile sets for ROI: ', roi));
    base_path = fullfile(data_dir, roi);

    n = 1;
    tile_size = tile_sizes(n);
    tile_size_m = tile_sizes_m(n);
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];

    save_path = fullfile(base_path, str);

    image = fullfile(data_dir, 'fixed', strcat('Bangalore_', roi, '.tif'));

    slumMask = fullfile(data_dir, 'masks', strcat('Bangalore_', roi, '_slumMask.tif'));
    otherMasks = {
        fullfile(data_dir, 'masks', strcat('Bangalore_',roi,'_urbanMask.tif'))
        fullfile(data_dir, 'masks', strcat('Bangalore_',roi,'_vegetationMask.tif'))
    };

    saved = slumTiling(image, tile_sizes, tile_step, factor, slumMask, save_path);
    nonSlumTiling(image, saved, tile_sizes, tile_step, factor, otherMasks, save_path);
end