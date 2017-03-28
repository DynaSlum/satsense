% multiclass_mask2oneclass_masks_Kalyan
% make multiple binary one-class masks from a multiclass mask for the
% 3 class classification problem: 1=BuiltUp, 2= NonBuildUp, 3 = Slum

%% parameters and filenames
saving = true;
visualizing = true;
overlay = true;

if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

data_path = fullfile(root_dir, 'Data','Kalyan','Rasterized_Lourens');
results_path = fullfile(root_dir, 'Results','Segmentation');
multiclass_mask = 'SegmentedImage_SURF_SVM_Classifier50_px100m80.mat';

image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';

result_slum_mask = 'slums_mask.tif';
result_builtup_mask = 'builtup_mask.tif';
result_nonbuiltup_mask = 'nonbuiltup_mask.tif';

image_fullfname = fullfile(data_path, image_fname);

result_builtup_mask_fullfname = fullfile(results_path, result_builtup_mask);
result_nonbuiltup_mask_fullfname = fullfile(results_path, result_nonbuiltup_mask);
result_slum_mask_fullfname = fullfile(results_path, result_slum_mask);

multiclass_mask_fullfname = fullfile(results_path, multiclass_mask);
%% load the data
image_data = imread(image_fullfname);
[nrows, ncols, ~] = size(image_data);

load(multiclass_mask_fullfname, 'segmented_image_denoised' );
multiclass_mask = segmented_image_denoised;
clear segmented_image_denoised;

%% make the 3 masks from the multiclass mask
[one_class_masks_array] = multiclass_mask2oneclass_masks(multiclass_mask);

%% assign each mask separately
builtup_mask = one_class_masks_array(:,:,1);
nonbuiltup_mask = one_class_masks_array(:,:,2);
slum_mask = one_class_masks_array(:,:,3);

%% save
if saving
    imwrite(builtup_mask, result_builtup_mask_fullfname);
    imwrite(nonbuiltup_mask, result_nonbuiltup_mask_fullfname);
    imwrite(slum_mask, result_slum_mask_fullfname);
end

%% visualize
if visualizing
    
    figure; imshow(logical(slum_mask)); title('Result Slums: Kalyan (cropped)');
    figure; imshow(logical(builtup_mask)); title('Result BuiltUp: Kalyan (cropped)');
    figure; imshow(logical(nonbuiltup_mask)); title('Result NonBuiltUp: Kalyan (cropped)');
    
    if overlay
        map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3
        red = cat(3, ones(nrows,ncols), zeros(nrows, ncols), zeros(nrows,ncols));
        green = cat(3, zeros(nrows,ncols), ones(nrows,ncols), zeros(nrows,ncols));
        blue = cat(3, zeros(nrows,ncols), zeros(nrows,ncols), ones(nrows,ncols));
        
        figure; imshow(image_data);
        hold on;
        hr=imshow(red);
        set(hr, 'AlphaData', 0.2*slum_mask);
        hg=imshow(green);
        set(hg, 'AlphaData', 0.2*nonbuiltup_mask);
        hb=imshow(blue);
        set(hb, 'AlphaData', 0.2*builtup_mask);
        hold off
        axis on, grid on;
        title('Result overlaid on image: Kalyan (cropped)');
        colormap(map);
        colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    end
end
