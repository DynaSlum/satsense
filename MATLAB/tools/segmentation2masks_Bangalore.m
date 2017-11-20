% segmentation2masks_Bangalore - generating masks per class for each class
% from the segmentation result for Bangalore ROIs


%% params
[ paths, processing_params, exec_flags] = config_params_Bangalore();

[data_dir, masks_dir, ~, segmentation_dir] = v2struct(paths);
[vocabulary_size, best_tile_size, best_tile_size_m, ~, ~, ~, ROIs] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

str = ['px' num2str(best_tile_size) 'm' num2str(best_tile_size_m)];

num_ROIs = length(ROIs);

for r = 1:num_ROIs
    roi = ROIs{r};
    
    if verbose
        disp(['Generating results masks for ', roi, '...']);
    end
    segm_fname = fullfile(segmentation_dir, ['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '_' roi '.mat']);

    %% load the data and the result    
    load(segm_fname, 'segmented_image_denoised'); % contains segmented_image (every 10th pixel); filled_segm_image and segmented_image_denoised
    [nrows, ncols, ~] = size(segmented_image_denoised);
    
    %% make 3 masks from the multiclass mask
    [masks] = multiclass_mask2oneclass_masks(segmented_image_denoised);
    
    %% save the binary masks to files
    slum_fname = fullfile(segmentation_dir, ['Bangalore_' roi '_slumResult.tif']);
    builtup_fname = fullfile(segmentation_dir, ['Bangalore_' roi '_builtupResult.tif']);
    nonbuiltup_fname = fullfile(segmentation_dir, ['Bangalore_' roi '_nonbuiltupResult.tif']);
    
    slum_mask = masks(:,:,3);
    builtup_mask = masks(:,:,1);
    nonbuiltup_mask = masks(:,:,2);
    
    imwrite(slum_mask, slum_fname);
    imwrite(builtup_mask, builtup_fname);
    imwrite(nonbuiltup_mask, nonbuiltup_fname);

end   
