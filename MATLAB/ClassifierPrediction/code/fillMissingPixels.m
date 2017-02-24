function [ filled_segm_image] = fillMissingPixels( sparse_segm_image, window_size)
%% fillMissingPixels  fills the non-processed (NaN value) pixels of a segmented 
% image represented as a sparse matrix. Each non-pocessed pixel gets a class
% determined in the following way: the most frequently occuring class in a window
% of size 4*tile_step (used to produce the sparse segmented image)

%   sparse_segm_image - the sparse segmented image with mostly NaN values
%                       and some classifier pixles, e.g. every tile_step ones
%   window_size - vector for 2 elements- the window size of rows and columns 
%               to define each pixels neighbourhood 
%   filled_segm_image- the filled segmented image 
% For Testing use test_fillMissingPixels

%% input control
if nargin < 2
    error('fillMissingPixels: not enough input arguments!');
end

%% params -> vars
% dimensions
nrows = size(sparse_segm_image, 1);
ncols = size(sparse_segm_image, 2);

%% initializations
filled_segm_image = sparse_segm_image;

%window_size = 4* tile_step;
half_window_size = fix(window_size/2); 

%% processing
% find indicies of the non-processed pixels
[nan_ind_r, nan_ind_c] = find(isnan(sparse_segm_image));
num_nan = length(nan_ind_r);
disp(['Number of unprocessed pixels: ', num2str(num_nan)]); 
disp('Press a key to continue...'); pause;

% loop over them
for ind_nan = 1 : num_nan
    r = nan_ind_r(ind_nan); c = nan_ind_c(ind_nan);
    
    sr = max(1, r - half_window_size(1));
    er = min(nrows, r + half_window_size(1) - 1);
   
    sc = max(1, c - half_window_size(2));
    ec = min(ncols, c + half_window_size(2) - 1);
    
    disp(['Processing pixel# ' num2str(ind_nan) ' out of ' num2str(num_nan) ' pixels...']); ;
    tic
        
        
    % get the data around that pixel
    data = sparse_segm_image(sr:er, sc:ec);  
    % make a vector
    data = data(:);
    
    % find the (first!) most frequent non NaN value
    [mfv, ~] = mode(data(~isnan(data)));
    
    % assing this value to the non-processed pixel
    filled_segm_image(r,c) = mfv;
    toc
end
        


