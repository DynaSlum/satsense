% oneclass_masks2multiclass_mask_Kalyan
% make a multiclass mask from the binary one-class masks for the
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
image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';
slum_mask = 'all_slums.tif';
builtup_mask = 'builtup_mask.tif';
nonbuiltup_mask = 'nonbuiltup_mask.tif';
multiclass_mask = 'all_classes_mask.tif';

image_fullfname = fullfile(data_path, image_fname);
builtup_mask_fullfname = fullfile(data_path, builtup_mask);
nonbuiltup_mask_fullfname = fullfile(data_path, nonbuiltup_mask);
slum_mask_fullfname = fullfile(data_path, slum_mask);
multiclass_mask_fullfname = fullfile(data_path, multiclass_mask);

%% load the data
image_data = imread(image_fullfname);
[nrows, ncols, ~] = size(image_data);
slum_mask = imread(slum_mask_fullfname);
builtup_mask = imread(builtup_mask_fullfname);
nonbuiltup_mask = imread(nonbuiltup_mask_fullfname);
% prepare masks array in the desired order
masks(:,:,1) = builtup_mask;
masks(:,:,2) = nonbuiltup_mask;
masks(:,:,3) = slum_mask;

%% make multiclass mask from th 3 masks
[multiclass_mask] = oneclass_masks2multiclass_mask(masks);

%% save
if saving
    map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3
    imwrite(multiclass_mask, map, multiclass_mask_fullfname);
end

%% visualize
if visualizing
    map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3
    RGB = ind2rgb(multiclass_mask, map);
    figure; imshow(RGB, map); title('Ground truth: Kalyan (cropped)');
    colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    axis on, grid on
    if overlay
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
        title('Ground truth overlaid on image: Kalyan (cropped)');
        colormap(map);
        colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    end
end
