% oneclass_masks2multiclass_mask_Bangalore
% make a multiclass mask from the binary one-class masks for the
% 3 class classification problem: 1=BuiltUp, 2= NonBuildUp, 3 = Slum

%% params
[ paths, processing_params, exec_flags] = config_params_Bangalore();

[~, masks_dir, ~, ~] = v2struct(paths);
[~, ~, ~, ~, ~, ~, ROIs] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

overlay = false;
fs = 16;

num_ROIs = length(ROIs);
%% displaying
for r = 1:num_ROIs
    roi = ROIs{r};
    
    if verbose
        disp(['Displaying ROI: ', roi, '...']);
    end
    
    
    %% load the mask data
    slum_mask = imread(fullfile(masks_dir,['Bangalore_' roi '_slumMask.tif']));
    [nrows, ncols] =size(slum_mask);
    masks = zeros(nrows, ncols, 3);
    slum_mask = slum_mask * 255;
    builtup_mask = imread(fullfile(masks_dir,['Bangalore_' roi '_urbanMask.tif']));
    % builtup_mask = builtup_mask * 255;
    nonbuiltup_mask = imread(fullfile(masks_dir,['Bangalore_' roi '_vegetationMask.tif']));
    % nonbuiltup_mask = nonbuiltup_mask * 255;
    
    %% saving mask filename
    multiclass_mask_fullfname = fullfile(masks_dir,['Bangalore_' roi '_allClassesMask.tif']);
    
    % prepare masks array in the desired order
    masks(:,:,1) = builtup_mask;
    masks(:,:,2) = nonbuiltup_mask;
    masks(:,:,3) = slum_mask;
    
    %% make multiclass mask from th 3 masks
    [multiclass_mask] = oneclass_masks2multiclass_mask(masks);
    
    %% save
    if sav
        map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3
        imwrite(multiclass_mask, map, multiclass_mask_fullfname);
    end
    
    %% visualize
    if visualize
        map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3
        RGB = ind2rgb(multiclass_mask, map);
        figure; imshow(RGB, map); %title('Ground truth: Bangalore');
       % colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
       % axis on, grid on
        handleToColorBar = colorbar('Ticks', [0.2 0.5 0.8]);
        set(handleToColorBar,'YTickLabel', []);
        hYLabel = ylabel(handleToColorBar,['BuiltUp        NonBuiltUp       Slum']);
        set(hYLabel,'Rotation',90);
        set(hYLabel,'FontSize',fs);
    end
    
end