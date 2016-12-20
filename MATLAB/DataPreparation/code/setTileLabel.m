function [class_label] = setTileLabel(tile_size, factor, tile_mask_slum, ...
            tile_mask_builtup, tile_mask_nonbuiltup)
%% setTileLabel  determine the tile class label given the corresponding tile class  masks
%   tile_size - vector for 2 elements- the number of rows and columns of the tile
%   factor - the factor for the minimum number of pixels from all pixels with given label
%   tile_mask_slum -the corrsponding mask tile with class label 'Slum'
%   tile_mask_builtup -the corrsponding mask tile with class label 'BuiltUp'
%   tile_mask_nonbuiltup -the corrsponding mask tile with class label 'NonBuiltUp'
%   Returns:
%   class_label - the tile class_label
% For Testing use test_setTileLabel

%% input control
if nargin < 4
    error('setTileLabel: 5 input parameters are expected!');
end

%% params -> vars
nrows = tile_size(1);
ncols = tile_size(2);

%% compute total number of pixels and# pixels per class
total_num = nrows * ncols;
num_slum = sum(tile_mask_slum(:));
num_builtup = sum(tile_mask_builtup(:));
num_nonbuiltup = sum(tile_mask_nonbuiltup(:));

%% find for which class is the maximum number of pixels
num_pixels = [num_slum num_builtup num_nonbuiltup];
[max_num, max_ind] = max(num_pixels);

%% decide on the final labeling
max_occurance = sum(num_pixels(:) == max_num);

if max_occurance > 1
    % we have more than one class_label with equal number of pixels
    class_label = 'Mixed';
else
    if max_num/total_num >= factor
        switch max_ind
            case 1
                class_label= 'Slum';
            case 2
                class_label = 'BuiltUp';
            case 3
                class_label = 'NonBuiltUp';
        end
    else
        % the number of that maximum class is not big enough
        class_label = 'Mixed';
    end
end
end
