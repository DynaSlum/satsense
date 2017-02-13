function [tile_names] = generateClassRandomTiles( image_fullfname, class_label, ...
    number_tiles, tile_size, factor, mask_fullfname, tiles_path)
%% generateClassRandomTiles - generating random tiles from a given class
%   image_fullname - the full name of the image to generate tiles from
%   class_label - the class from which we want randowm tiles
%   number_tiles - the desired number of tiles 
%   tile_size - vector for 2 elements- the number of rows and columns of the tile
%   factor - the factor for the minimum number of pixels from all pixels with given label
%   mask_fullfname - the full names of the binary mask corresponding to
%                    the class label (1:'BuiltUp', 2: 'NonBuiltUp' or 3: 'Slum')
%   tiles_path - the main filder for the tiles to be saved in subfolders
%
%   tiles - the full filenames of the tiles
% For Testing use test_generateRandomTiles. See also nonSlumTiling.m

%% input control
if nargin < 7
    error('nonSlumTiling: not enough input arguments!');
end

%% params -> vars

% basename and extention for the tiles
[~,base_fname,ext] = fileparts(image_fullfname);
ext = ext(2:end);

%% initializations
tile_names = {};

%% load the data from the files
image_data = imread(image_fullfname);
mask = imread(mask_fullfname);
[nrows, ncols] = size(mask);


% f = figure;
% subplot(221); imshow(image_data);axis on;grid on;title('Image data');
% subplot(222); imshow(mask);axis on;grid on;title(['Mask for ' class_label ' class']);

%% dimensions
nrows_tile = tile_size(1); ncols_tile = tile_size(2);
half_nrows_tile = fix(nrows_tile/2); half_ncols_tile = fix(ncols_tile/2);

%% generate valid 'strict' mask locations and an image tile
i = 0;
total_num = nrows_tile * ncols_tile;
%rng(0,'twister');

while i < number_tiles
    
    % generate random pixels
    min_r = half_nrows_tile + 1;
    max_r = nrows - min_r;
    pr = fix((max_r-min_r)*rand + min_r);
       
    min_c = half_ncols_tile + 1;
    max_c = ncols - min_c;
    pc = fix((max_c-min_c)*rand + min_c);
    
    % tile extent
    sr = pr - half_nrows_tile;
    er = sr + nrows_tile -1;
    sc = pc - half_ncols_tile;
    ec = sc + ncols_tile - 1;
    
    % check if the central pixel of a tile belongs to a mask
    if mask(pr, pc)
        % get the mask tile
        tile_mask  = mask(sr:er, sc:ec);
        % determine if the tile satisfies the strict condition
        num_class = sum(tile_mask(:));
        % if it is mark the location in the strict mask
        if num_class/total_num >= factor
            i = i + 1;                                   
            extent = [sr er sc ec];
            image_tile  = image_data(sr:er, sc:ec, :);
            % save
            tile_name = saveTile2File(image_tile, extent, tiles_path,...
                class_label, base_fname, ext);
            tile_names{i} = tile_name;
        end
    end
end



