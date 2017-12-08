
%% params
[ paths, processing_params, exec_flags] = config_params_Bangalore();

[data_dir, masks_dir, ~, ~] = v2struct(paths);
[~, ~, ~, ~, ~, ~, ROIs] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

num_ROIs = length(ROIs);

%% displaying
for r = 2 %1:num_ROIs
    roi = ROIs{r};
    
    if verbose
        disp(['Displaying ROI: ', roi, '...']);
    end
    
    image_data = imread(fullfile(data_dir,['Bangalore_' roi '.tif']));
    
    nrows = size(image_data,1);
    ncols = size(image_data,2);
    
    %% load the mask data
    slum_mask = imread(fullfile(masks_dir,['Bangalore_' roi '_slumMask.tif']));
    slum_mask = slum_mask * 255;
    builtup_mask = imread(fullfile(masks_dir,['Bangalore_' roi '_urbanMask.tif']));
   % builtup_mask = builtup_mask * 255;
    nonbuiltup_mask = imread(fullfile(masks_dir,['Bangalore_' roi '_vegetationMask.tif']));
   % nonbuiltup_mask = nonbuiltup_mask * 255;
    %% prepare colored overlays
    red = cat(3, ones(nrows,ncols), zeros(nrows, ncols), zeros(nrows,ncols));
    green = cat(3, zeros(nrows,ncols), ones(nrows,ncols), zeros(nrows,ncols));
    blue = cat(3, zeros(nrows,ncols), zeros(nrows,ncols), ones(nrows,ncols));
    
%     %% display the image daya
%     figure;imshow(image_data);
%     axis on, grid on;
%     title('Image data');
%     
    %% display the image daya and overlap the overlays
    figure; imshow(image_data);
    hold on;
    hr=imshow(red);
    set(hr, 'AlphaData', 0.2*slum_mask);
    hg=imshow(green);
    set(hg, 'AlphaData', 0.2*nonbuiltup_mask);
    hb=imshow(blue);
    set(hb, 'AlphaData', 0.2*builtup_mask);
    hold off
    map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3
    
    %axis on, grid on;
    %title('Ground truth overalyed on Kalyan cropped image');
    colormap(map);
    %colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'}, 'FontSize',26);
    handleToColorBar = colorbar('Ticks', [0.2 0.5 0.8]);
    set(handleToColorBar,'YTickLabel', []);
    hYLabel = ylabel(handleToColorBar,['BuiltUp           NonBuiltUp          Slum']);     
    set(hYLabel,'Rotation',90);
    set(hYLabel,'FontSize',26);
    
end
