function [imds] = createImageDatastore( image_dataset_location, summary_flag,...
    preview_flag)

%% createImageDatastore  wrapper aroind imageDatastore for datasets with 
%   some preset parameters
%   INPUT:
%   image_dataset_location -top folder of the images. Each subfolder
%       corresponds to 4 classes with class labels 1:'Slum', 2: 'BuiltUp' 
%       and 3: 'NonBuiltUp', 4: 'Mixed'. The funciton uses as default 
%       settings for imageDatastore parameters as follows: 
%       'IncludeSubfolders' = true and 'LabelSource' = 'foldernames'
%   summary_flag -if true class names and counts are displayed
%   preview_flag - if true mosaic of sample for classes is shown 
%   OUPUT:
%   imgs - returns image datastore object
%   optionally displays class names and counts and shows mosaic of class
%   samples
% For Testing use test_createImageDatastore
% Note: see also https://nl.mathworks.com/matlabcentral/fileexchange/...
%   58320-demos-from--object-recognition--deep-learning--webinar/content/...
%   DeepLearningWebinar/Demo1_BagOfFeatures/Scene_Identification.m
%% create a new datastore
imds = imageDatastore(image_dataset_location,...
    'IncludeSubfolders',true,'LabelSource','foldernames')              %#ok


%% Display Class Names and Counts
if summary_flag || preview_flag
    tbl = countEachLabel(imds);                     %#ok
end

if summary_flag    
    disp(tbl);
end

%% Show sampling of all data
sample = splitEachLabel(imds,16);

if preview_flag
    for ii = 1:3
        sf = (ii-1)*16 +1;
        ax(ii) = subplot(2,2,ii);
        montage(sample.Files(sf:sf+3));
        title(char(tbl.Label(ii)));
    end
end