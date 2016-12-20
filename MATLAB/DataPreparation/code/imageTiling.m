function [ ] = imageTiling( image_fullfname, tile_size, tile_step, factor, ...
    masks_fullfnames, tiles_path)
%% imageTiling  cropping an image to tiles and saving them to files
%   The fucntion crops a given image and several class masks to image tiles
%   which are saved as image files in folders corresponding to the masks
%   image_fullname - the full name of the image to be croped/tiles
%   tile_size - vector for 2 elements- the number of rows and columns of the tile
%   tile_step - vector for 2 elements- the step size of rows and columns 
%   factor - the factor for the minimum number of pixels from all pixels with given label
%   masks_fullfnames -cel array of the full names of the binary masks corresponding to
%                     the class label 1:'Slum', 2: 'BuiltUp' and 3: 'NonBuiltUp'
%   tiles_path - the main filder for the tiles to be saved in subfolders
% For Testing use test_imageTiling

%% input control
if nargin < 6
    error('imageTiling: not enough input arguments!');
end

%% params -> vars
slum_mask_fname = char(masks_fullfnames{1});
builtup_mask_fname = char(masks_fullfnames{2});
nonbuiltup_mask_fname = char(masks_fullfnames{3});

% basename and extention for the tiles
[~,base_fname,ext] = fileparts(image_fullfname);  
ext = ext(2:end);

%% load the data from the files
image_data = imread(image_fullfname);
slum_mask = imread(slum_mask_fname);
builtup_mask = imread(builtup_mask_fname);
nonbuiltup_mask = imread(nonbuiltup_mask_fname);

%% dimensions
nrows = size(image_data, 1); ncols = size(image_data, 2); 
nrows_tile = tile_size(1); ncols_tile = tile_size(2);
nrows_step = tile_step(1); ncols_step = tile_step(2);

%% tiling
for sr = 1: nrows_step : nrows
    er = min(nrows, sr + nrows_tile - 1);
    for sc = 1: ncols_step : ncols
        ec = min(ncols, sc + ncols_tile - 1);
        extent = [sr er sc ec];
        
        % adjust the tile size if necessary
        real_tile_size(1) = er - sr + 1;
        real_tile_size(2) = ec - sc + 1;
        
        if (real_tile_size(1) < tile_size(1)) 
            tile_size(1) = real_tile_size(1);
        else
            tile_size(1) = nrows_tile;
        end
        if (real_tile_size(2) < tile_size(2)) 
            tile_size(2) = real_tile_size(2);
        else
            tile_size(2) = ncols_tile;
        end
        
        % get the image and masks tile
        image_tile = image_data(sr:er, sc:ec, :);
        slum_tile  = slum_mask(sr:er, sc:ec,:);
        builup_tile = builtup_mask(sr:er, sc:ec,:);
        nonbuiltup_tile = nonbuiltup_mask(sr:er, sc:ec,:);
        
        % determine the class label for the tile
        label = setTileLabel(tile_size, factor, ...
        slum_tile, builup_tile, nonbuiltup_tile);
        
        % save the tile at the location given by the path and label
        saveTile2File(image_tile, extent, tiles_path, label, base_fname, ext);
    end
end

end

