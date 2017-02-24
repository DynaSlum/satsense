function [ filt_segm_image] = majorityFilterSegmentation( segm_image, window_size)
%% majorityFilterSegmentation  denoises a segmentation using majority filter
%   segm_image - the noisy segmented image 
%   window_size - vector for 2 elements- the window size of rows and columns 
%               to define each pixels neighbourhood 
%   filt_segm_image- the filtered segmented image 
% For Testing use test_majorityFilterSegmentation

%% input control
if nargin < 2
    error('majorityFilterSegmentation: not enough input arguments!');
end


%% processing
fun = @(block_struct) mode(block_struct.data(:)) * ones(size(block_struct.data)); 
filt_segm_image = blockproc(segm_image, window_size,fun);
        


