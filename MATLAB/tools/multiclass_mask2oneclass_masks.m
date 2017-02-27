% multiclass_mask2oneclass_masks - function to make binary multiclass masks from
% a multiclass mask
% multiclass_mask - the input milticlass mask, where each class is encoded
% by a separate number
% one_class_masks_array- the output one-class masks stacked in a 3D array in
% the order they should be encoded in the output multi-class mask. For
% example for the DynaSlum 3 class classification the convention is
% 1=BuiltUp, 2= NonBuildUp, 3 = Slum

function [one_class_masks_array] = multiclass_mask2oneclass_masks(multiclass_mask)

%% dimensions
[nrows, ncols] = size(multiclass_mask);

nclasses = max(max(multiclass_mask));
%% init
one_class_masks_array = zeros(nrows, ncols, nclasses);

%% summarize the info from many masks to 1 mask
for nc = 1: nclasses
    [rows,cols] = find(multiclass_mask==nc);
    for i = 1:length(rows)
        r = rows(i); c = cols(i);
        one_class_masks_array(r,c,nc) = 1;
    end
end
