% show_image_overlaid_segmentation - generating a figure with overlaied
% segmentation on it


%% parameters and filenames
vis_mask_only = false;
overlay = true;
mapw = [0 0 1; 0 1 0; 1 0 0; 1 1 1]; % Blue, Green, Red, White = 1,2,3, NaN
map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3

if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

data_path = fullfile(root_dir, 'Data','Kalyan','Rasterized_Lourens');
image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';
image_fullfname = fullfile(data_path, image_fname);
results_path = fullfile(root_dir, 'Results', 'Segmentation');
segm_fname = fullfile(results_path, 'SegmentedImage_SURF_SVM_Classifier50_px100m80.mat');

%% load the data and the result
image_data = imread(image_fullfname);
[nrows, ncols, ~] = size(image_data);
load(segm_fname); % contains segmented_image (every 5th pixel); filled_segm_image and segmented_image_denoised


%% visualize
if vis_mask_only
    RGB1 = ind2rgb(segmented_image, mapw);
    figure; imshow(RGB1, mapw); title('Segmented Kalyan cropped image');
    xlabel('Every 5th pixel is processed');
    colorbar('Ticks', [0.1 0.35 0.65 0.9], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum', 'Not Processed'});
    axis on, grid on
end

if overlay
    %% make 3 masks from the multiclass mask
    [masks] = multiclass_mask2oneclass_masks(segmented_image);
    red = cat(3, ones(nrows,ncols), zeros(nrows, ncols), zeros(nrows,ncols));
    green = cat(3, zeros(nrows,ncols), ones(nrows,ncols), zeros(nrows,ncols));
    blue = cat(3, zeros(nrows,ncols), zeros(nrows,ncols), ones(nrows,ncols));
    
    figure; imshow(image_data);
    hold on;
    hr=imshow(red);
    set(hr, 'AlphaData', 0.2*masks(:,:,3));
    hg=imshow(green);
    set(hg, 'AlphaData', 0.2*masks(:,:,2));
    hb=imshow(blue);
    set(hb, 'AlphaData', 0.2*masks(:,:,1));
    hold off
    axis on, grid on;
    title('Segmentation overlaid on Kalyan cropped image');
    colormap(mapw);
    
end

if vis_mask_only
    RGB2 = ind2rgb(filled_segm_image, map);
    figure; imshow(RGB2, map); title('Filled segmented Kalyan cropped image');
    xlabel('Missing pixels filled with majority vote fom a 20x20 window');
    colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    axis on, grid on
end

if overlay
    %% make 3 masks from the multiclass mask
    [masks] = multiclass_mask2oneclass_masks(filled_segm_image);
    red = cat(3, ones(nrows,ncols), zeros(nrows, ncols), zeros(nrows,ncols));
    green = cat(3, zeros(nrows,ncols), ones(nrows,ncols), zeros(nrows,ncols));
    blue = cat(3, zeros(nrows,ncols), zeros(nrows,ncols), ones(nrows,ncols));
    
    figure; imshow(image_data);
    hold on;
    hr=imshow(red);
    set(hr, 'AlphaData', 0.2*masks(:,:,3));
    hg=imshow(green);
    set(hg, 'AlphaData', 0.2*masks(:,:,2));
    hb=imshow(blue);
    set(hb, 'AlphaData', 0.2*masks(:,:,1));
    hold off
    axis on, grid on;
    title('Filled segmentation overlaid on Kalyan cropped image');
    colormap(map);
    colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    
end

if vis_mask_only
    RGB3 = ind2rgb(segmented_image_denoised, map);
    figure; imshow(RGB3, map); title('Denoised segmented Kalyan cropped image');
    xlabel('Majority filter of size 40x40 is used');
    colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    axis on, grid on
end

if overlay
    %% make 3 masks from the multiclass mask
    [masks] = multiclass_mask2oneclass_masks(segmented_image_denoised);
    red = cat(3, ones(nrows,ncols), zeros(nrows, ncols), zeros(nrows,ncols));
    green = cat(3, zeros(nrows,ncols), ones(nrows,ncols), zeros(nrows,ncols));
    blue = cat(3, zeros(nrows,ncols), zeros(nrows,ncols), ones(nrows,ncols));
    
    figure; imshow(image_data);
    hold on;
    hr=imshow(red);
    set(hr, 'AlphaData', 0.2*masks(:,:,3));
    hg=imshow(green);
    set(hg, 'AlphaData', 0.2*masks(:,:,2));
    hb=imshow(blue);
    set(hb, 'AlphaData', 0.2*masks(:,:,1));
    hold off
    axis on, grid on;
    title('Denoised segmentation overlaid on Kalyan cropped image');
    colormap(map); 
    colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    
end

