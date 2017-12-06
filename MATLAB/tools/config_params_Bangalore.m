%[ paths, processing_params, exec_flags] = config_params_Bangalore()
% config_params_Bangalore - parameter configurations for the Bangalore scripts
% **************************************************************************
% author: Elena Ranguelova, NLeSc
% date created: 3 November 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last modification date: 8 Nov 2017
% modification details: added window_sizes for filling the missing pixels
% and for the majority filter
%**************************************************************************
% INPUTS:
%**************************************************************************
% OUTPUTS:
% paths
% data_dir - the folder containing the image data
% classifier_dir - the folder containing the trained classifier
% ...
%**************************************************************************
function [ paths, processing_params, exec_flags] = config_params_Bangalore()
    
%% paths
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end
data_root_dir = fullfile(root_dir, 'Data','Bangalore','GEImages');
results_dir = fullfile(root_dir, 'Results','Bangalore');

classifier_dir = fullfile(results_dir, 'Classification3Classes', 'Classifiers');
segmentation_dir = fullfile(results_dir, 'Segmentation');
    
data_dir = fullfile(data_root_dir, 'Clipped','fixed');
masks_dir = fullfile(data_root_dir, 'masks'); 

paths = v2struct(data_dir, masks_dir, classifier_dir, segmentation_dir);

%% processing params
vocabulary_size = [50];
best_tile_size = [268];
best_tile_size_m = [40];
stepY = 10;
stepX = stepY;
tile_step = [stepX stepY];
wsY = 22;wsX = wsY;
window_size_miss = [wsX wsY];
wsY = 30;wsX = wsY;
window_size_filt = [wsX wsY];

ROIs = {
    'ROI1'
    'ROI2'
    'ROI3'
    'ROI4'
    'ROI5'
     };
processing_params = v2struct(vocabulary_size, best_tile_size, ...
    best_tile_size_m, tile_step, window_size_miss, window_size_filt, ROIs);

%% exec_params
verbose = true;
visualize = true;
sav = true;

exec_flags = v2struct(verbose, visualize, sav);