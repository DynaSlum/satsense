function [ number_images] = slumTiling( image_fullfname, tile_size, tile_step, factor, ...
    mask_fullfname, tiles_path)
%% slumTiling  cropping the part of the image lebelled as slump to tiles and saving them to files
%   image_fullname - the full name of the image to be croped/tiles
%   tile_size - vector for 2 elements- the number of rows and columns of the tile
%   tile_step - vector for 2 elements- the step size of rows and columns
%   factor - the factor for the minimum number of pixels from all pixels with given label
%   mask_fullfname - the full name of the binary mask corresponding to
%                     the class label 'Slum'
%   tiles_path - the main filder for the tiles to be saved in subfolders
%   save_mixed - flag indicating whether to save tiles of class Mixed. default is false.
%
%   number_images- the number of saved Slum images/tiles
% For Testing use test_slumTiling. See also imageTiling.m

%% input control
if nargin < 6
    error('slumTiling: not enough input arguments!');
end

%% params -> vars

% basename and extention for the tiles
[~,base_fname,ext] = fileparts(image_fullfname);
ext = ext(2:end);

%% initializations
number_images = 0;

%% load the data from the files
image_data = imread(image_fullfname);
slum_mask = imread(mask_fullfname);

%% dimensions
nrows = size(image_data, 1); ncols = size(image_data, 2);
nrows_tile = tile_size(1); ncols_tile = tile_size(2);
half_nrows_tile = fix(nrows_tile/2); half_ncols_tile = fix(ncols_tile/2);
nrows_step = tile_step(1); ncols_step = tile_step(2);

total_num = nrows_tile * ncols_tile;
%% tiling
for pr = half_nrows_tile + 1: nrows_step : nrows - half_nrows_tile
    for pc = half_ncols_tile + 1: ncols_step : ncols - half_ncols_tile
        
        % check if the centralpixelof a tile belongs to a slum and only
        % then proceed
        if slum_mask(pr, pc)
            sr = pr - half_nrows_tile;
            er = sr + nrows_tile - 1;
            sc = pc - half_ncols_tile;
            ec = sc + ncols_tile - 1;
            
            extent = [sr er sc ec];
            
            % get the mask tile
            slum_tile  = slum_mask(sr:er, sc:ec, :);
            
            % determine if the tile should be considered for training slums           
            num_slum = sum(slum_tile(:));
            
            
            % if it is save the image tile to file
            if num_slum/total_num >= factor
                number_images = number_images + 1;
                image_tile = image_data(sr:er, sc:ec, :);
                saveTile2File(image_tile, extent, tiles_path, 'Slum', base_fname, ext);
            end
        end
    end
end

