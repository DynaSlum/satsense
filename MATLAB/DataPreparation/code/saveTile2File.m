function [filename] = saveTile2File(image_data,  extent, path, class_label, base_fname, ext)
%% saveTile2File  saving image tile (patch) to file
%   image_data - 3D array of image data values
%   extent - vector of 4 elements, starting and ending row and column indicies
%   path - dataset path
%   class_label- the tile/patch class label. Can be one of "Slum",
%               "BuiltUp" or "NonBuiltUp"
%   base_fname - base file name
%   ext - filename extention
% For Testing use test_saveTile2File

%% input control
if not(ismember(class_label,{'BuiltUp','NonBuiltUp','Slum','Mixed'}))
    error('saveTile2File: unknown class label!');
end

if length(extent) ~= 4
    error('saveTile2File: the tile extent should have 4 elements- starting and ending row and column index!');
end

%% params -> vars
sr = extent(1);
er = extent(2);
sc = extent(3);
ec = extent(4);

%% generate the filename
fname = [base_fname '_tile_' 'sr' num2str(sr) 'er' num2str(er) ...
    'sc' num2str(sc) 'ec' num2str(ec) '.' ext];
% if the sub-dir doesn't exist creat eit
full_path = fullfile(path, class_label);
if ~isdir(full_path)
    mkdir(full_path);
end
filename = fullfile(full_path,fname);

%% save at the specified location
imwrite(image_data, filename, ext);

end
