% oneclass_masks2multiclass_mask - function to make a multiclass mask from
% binary one-class masks
% one_class_masks_array- the input one-class masks stacked in a 3D array in
% the order they should be encoded in the output multi-class mask. For
% example for the DynaSlum 3 class classification the convention is
% 1=BuiltUp, 2= NonBuildUp, 3 = Slum
% multiclass_mask - the outpul milticlass mask, where each class is encoded
% by a separate number

function [multiclass_mask] = oneclass_masks2multiclass_mask(one_class_masks_array)

%% dimensions
[nrows, ncols, nclasses] = size(one_class_masks_array);

%% init
multiclass_mask = zeros(nrows, ncols);

%% summarize the info from many masks to 1 mask
for nc = 1: nclasses
    class_ind = find(one_class_masks_array(:,:,nc));
    multiclass_mask(class_ind) = nc;
end
