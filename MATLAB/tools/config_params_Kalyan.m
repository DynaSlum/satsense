%[ paths, processing_params, exec_flags] = config_params_Kalyan()
% config_params_Kalyan - parameter configurations for the Kalyan scripts
% **************************************************************************
% author: Elena Ranguelova, NLeSc
% date created: 6 December 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last modification date: 
% modification details: 
%**************************************************************************
% INPUTS:
%**************************************************************************
% OUTPUTS:
% paths
% data_dir - the folder containing the image data
% classifier_dir - the folder containing the trained classifier
% ...
%**************************************************************************
function [ paths, processing_params, exec_flags] = config_params_Kalyan()
    
%% paths
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end
data_root_dir = fullfile(root_dir, 'Data','Kalyan');
results_dir = fullfile(root_dir, 'Results','Kalyan');

datastores_dir = fullfile(results_dir, 'Classification3Classes', 'DatastoresAndFeatures');
classifier_dir = fullfile(results_dir, 'Classification3Classes', 'Classifiers');
performance_dir = fullfile(results_dir, 'Classification3Classes', 'Performance');

segmentation_dir = fullfile(results_dir, 'Segmentation');
    
data_dir = fullfile(data_root_dir, 'Data4Matlab3ClassExperiment');
masks_dir = fullfile(data_root_dir, 'masks','threshold_13'); 

paths = v2struct(data_dir, masks_dir, datastores_dir, classifier_dir, performance_dir, segmentation_dir);

roi = 'ROI1';
%% processing params
vocabulary_size = [];
best_tile_size = [];
best_tile_size_m = [];
stepY = 0;
stepX = stepY;
tile_step = [stepX stepY];
wsY = 0;wsX = wsY;
window_size_miss = [wsX wsY];
wsY = 0;wsX = wsY;
window_size_filt = [wsX wsY];


processing_params = v2struct(vocabulary_size, best_tile_size, ...
    best_tile_size_m, tile_step, window_size_miss, window_size_filt, roi);

%% exec_params
verbose = true;
visualize = true;
sav = true;

exec_flags = v2struct(verbose, visualize, sav);