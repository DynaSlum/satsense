function [ segmented_image] = classifyEachPixelFromTile( image_fullfname, tile_size, tile_step, classifier)
%% classifyEachPixelFromTile  classifies each pixel of the original image based on tiles around the pixels
%   The function classifies each pixel of a given image using a tile around
%   it given a pre-trained classifer on tiles

%   image_fullname - the full name of the image to be croped/tiles
%   tile_size - vector for 2 elements- the number of rows and columns of the tile
%   tile_step - vector for 2 elements- the step size of rows and columns 
%   classifier - 
%    
%   segmented_image- the segmented image having values 3 for Slum (R), 2
%   for NonBuiltUp (G) and 1- BuiltUp (B)
% For Testing use test_classifyEachPixelFromTile

%% input control
if nargin < 4
    error('classifyEachPixelFromTile: not enough input arguments!');
end

%% params -> vars
% basename and extention for the tiles
[~,base_fname,ext] = fileparts(image_fullfname);  
ext = ext(2:end);


%% load the data from the input file
image_data = imread(image_fullfname);

%% dimensions
nrows = size(image_data, 1);
ncols = size(image_data, 2);

%% initializations
segmented_image = NaN(nrows, ncols);

nrows_tile = tile_size(1); ncols_tile = tile_size(2);
half_nrows_tile = fix(nrows_tile/2);  half_ncols_tile = fix(ncols_tile/2); 
nrows_step = tile_step(1); ncols_step = tile_step(2);

%% tiling
for r = 1: nrows_step : nrows
    
    sr = max(1, r - half_nrows_tile);
    er = min(nrows, r + half_nrows_tile - 1);
    for c = 1: ncols_step : ncols
       
        sc = max(1, c - half_ncols_tile);
        ec = min(ncols, c + half_ncols_tile - 1);
        disp('Processing pixel :'); pixel = [r c]
        
        
        % get the image tile
        image_tile = image_data(sr:er, sc:ec, :);                
        
        % classify the image using the classifier
        tic
        [labelIdx, scores] = predict(classifier, image_tile);
        toc
        class_label = char(classifier.Labels(labelIdx))
        
        switch class_label
            case 'Slum'
                l = 3;
            case 'NonBuiltUp'
                l = 2;
            case 'BuiltUp'
                l = 1;
        end
        
        segmented_image(r,c) = l;                         
    end
end

end

