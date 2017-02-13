function [number_images] = nonSlumTiling( image_fullfname, number_tiles,...
    tile_size, tile_step, factor, masks_fullfnames, tiles_path)
%% nonSlumTiling  cropping the part of the image labelled as Non slum  (BuiltUp or Non BuildUp) to tiles and saving them to files
%   image_fullname - the full name of the image to be croped/tiles
%   number_tiles - the desirednumber of tiles (obtained from slumTiling.m
%   tile_size - vector for 2 elements- the number of rows and columns of the tile
%   tile_step - vector for 2 elements- the step size of rows and columns
%   factor - the factor for the minimum number of pixels from all pixels with given label
%   masks_fullfnames - cel array of the full names of the binary masks corresponding to
%                      the class labels 1:'BuiltUp' and 2: 'NonBuiltUp'
%   tiles_path - the main filder for the tiles to be saved in subfolders
%
%   number_images- structure containing the number of images per each class:
%                  BuildUp|NonBuildUp
% For Testing use test_nonSlumTiling. See also imageTiling.m and slumTiling.m

%% input control
if nargin < 7
    error('nonSlumTiling: not enough input arguments!');
end

%% params -> vars
builtup_mask_fname = char(masks_fullfnames{1});
nonbuiltup_mask_fname = char(masks_fullfnames{2});

% basename and extention for the tiles
[~,base_fname,ext] = fileparts(image_fullfname);
ext = ext(2:end);

%% initializations
number_images.builtup = 0;
number_images.nonbuiltup = 0;

%% load the data from the files
image_data = imread(image_fullfname);
builtup_mask = imread(builtup_mask_fname);
nonbuiltup_mask = imread(nonbuiltup_mask_fname);

% f = figure;
% subplot(221); imshow(builtup_mask);axis on;grid on;title('BuiltUp mask');
% subplot(223); imshow(nonbuiltup_mask);axis on;grid on;title('NonBuiltUp mask');
%% dimensions
nrows = size(image_data, 1);
ncols = size(image_data, 2);
nrows_tile = tile_size(1); ncols_tile = tile_size(2);
half_nrows_tile = fix(nrows_tile/2); half_ncols_tile = fix(ncols_tile/2);
nrows_step = tile_step(1); ncols_step = tile_step(2);

%% initialization
builtup_mask_strict = zeros(nrows, ncols);
nonbuiltup_mask_strict = zeros(nrows, ncols);

total_num = nrows_tile * ncols_tile;
%% make new 'strict' masks containing only pixels which fit
for pr = half_nrows_tile + 1: nrows_step: nrows - half_nrows_tile
    for pc = half_ncols_tile + 1:ncols_step: ncols - half_ncols_tile
        %pr
        %pc
        % tile extent
        sr = pr - half_nrows_tile;
        er = sr + nrows_tile -1;
        sc = pc - half_ncols_tile;
        ec = sc + ncols_tile - 1;
        
        % check if the central pixel of a tile belongs to a mask
        if builtup_mask(pr, pc)
            % get the mask tile
            builtup_tile  = builtup_mask(sr:er, sc:ec, :);
            % determine if the tile should be considered for training
            num_builtup = sum(builtup_tile(:));
            % if it is markthe location in the strict mask
            if num_builtup/total_num >= factor
                builtup_mask_strict(pr,pc) = 1;
            end
        elseif nonbuiltup_mask(pr, pc)
            % get the mask tile
            nonbuiltup_tile  = nonbuiltup_mask(sr:er, sc:ec, :);
            % determine if the tile should be considered for training
            num_nonbuiltup = sum(nonbuiltup_tile(:));
            % if it is markthe location in the strict mask
            if num_nonbuiltup/total_num >= factor
                nonbuiltup_mask_strict(pr,pc) = 1;
            end
        end
    end
end
% figure(f);
% subplot(222); imshow(builtup_mask_strict);axis on, grid on, title('BuiltUp mask strict');
% subplot(224); imshow(nonbuiltup_mask_strict);axis on, grid on, title('NonBuiltUp mask strict');
%
% number_images.builtup = sum(builtup_mask_strict(:));
% number_images.nonbuiltup = sum(nonbuiltup_mask_strict(:));

%% get given number of tiles from both classes
% BuiltUp
idx_bu = find(builtup_mask_strict == 1);
perm = randperm(length(idx_bu), number_tiles);
rand_idx_bu =  idx_bu(perm);
[prows_bu, pcols_bu] = ind2sub([nrows, ncols], rand_idx_bu);


% NonBuiltUp
idx_nbu = find(nonbuiltup_mask_strict == 1);
perm = randperm(length(idx_nbu), number_tiles);
rand_idx_nbu =  idx_nbu(perm);
[prows_nbu, pcols_nbu] = ind2sub([nrows, ncols], rand_idx_nbu);

%% tiling
% BuiltUp
for n = 1 : number_tiles
    pr = prows_bu(n); pc = pcols_bu(n);
    sr = pr - half_nrows_tile;
    er = sr + nrows_tile - 1;
    sc = pc - half_ncols_tile;
    ec = sc + ncols_tile - 1;
    
    extent_bu = [sr er sc ec];
    
    % get the mask tile
    number_images.builtup = number_images.builtup + 1;
    image_tile  = image_data(sr:er, sc:ec, :);
    % save
    [~] = saveTile2File(image_tile, extent_bu, tiles_path, 'BuiltUp', base_fname, ext);
    
end

% NonBuiltUp
for n = 1 : number_tiles
    pr = prows_nbu(n); pc = pcols_nbu(n);
    sr = pr - half_nrows_tile;
    er = sr + nrows_tile - 1;
    sc = pc - half_ncols_tile;
    ec = sc + ncols_tile - 1;
    
    extent_nbu = [sr er sc ec];
    
    % get the mask tile
    number_images.nonbuiltup = number_images.nonbuiltup + 1;
    image_tile  = image_data(sr:er, sc:ec, :);
    % save
    [~] = saveTile2File(image_tile, extent_nbu, tiles_path, 'NonBuiltUp', base_fname, ext);
    
end



