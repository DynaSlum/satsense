function [filename] = saveTile2File(image_data,  extent, path, class_label, base_fname, ext)
%% saveTile2File  saving image tile (patch) to file
%   image_data - 3D array of image data values
%   extent - either vector of 4 elements, starting and ending row and
%            column indicies OR vector with 2 elements- row and column
%   path - dataset path
%   class_label- the tile/patch class label. Can be one of "Slum",
%               "BuiltUp" or "NonBuiltUp"
%   base_fname - base file name
%   ext - filename extention
% For Testing use test_saveTile2File

%% input control
if not(ismember(class_label,{'','BuiltUp','NonBuiltUp','Slum','Mixed'}))
    error('saveTile2File: unknown class label!');
end

if length(extent) ~= 4 && length(extent) ~= 2
    error('saveTile2File: the tile extent should have 4 elements- starting and ending row and column index or 2 elements- pixel row and column!');
end

%% params -> vars
if length(extent) == 2
    pixel_wise = true;
elseif length(extent) == 4
    pixel_wise = false;
end

if pixel_wise
    r = extent(1);
    c = extent(2);
else
    sr = extent(1);
    er = extent(2);
    sc = extent(3);
    ec = extent(4);
end

%% generate the filename
if pixel_wise
    fname = [base_fname '_tile_' 'r' num2str(r) ...
        'c' num2str(c) '.' ext];
    full_path = path;
else
    
    fname = [base_fname '_tile_' 'sr' num2str(sr) 'er' num2str(er) ...
        'sc' num2str(sc) 'ec' num2str(ec) '.' ext];
    % if the sub-dir doesn't exist create it
    full_path = fullfile(path, class_label);
end
if ~isdir(full_path)
    mkdir(full_path);
end
filename = fullfile(full_path,fname);

%% save at the specified location
imwrite(image_data, filename, ext);

end
