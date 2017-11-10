
%% params
[ paths, processing_params, exec_flags] = config_params_Bangalore();

[data_dir, masks_dir, ~, ~] = v2struct(paths);
[~, ~, ~, ~, ~, ~, ROIs] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

num_ROIs = length(ROIs);

%% displaying
for r = 3 %1:num_ROIs
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
    %% add special rectangles of interest
    switch r
        case 1
            rec_pos_g = [3534 4036 446 329; 4694 4005 255 131;...
                4243 4518 243 423; 4527 4488 427 263;
                3312 4685 115 123];
            labels_str = {'GT1'; 'GT2'; 'GT3'; 'GT4'; 'GT5'};
            image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_g,...
                        labels_str, 'Font', 'LucidaTypewriterBold', 'FontSize',42, ...
                        'TextColor','black', 'Color', 'green', 'TextBoxOpacity',0.7,'LineWidth',12);
            rec_pos_r = [3801 5023 108 71; 4886 5051 87 47];
            labels_str = {'GT6';'GT7'};
            image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_r,...
            labels_str, 'Font', 'LucidaTypewriterBold','FontSize',36,'TextColor','white',...
                        'Color', 'red', 'TextBoxOpacity',0.7,'LineWidth',6);
        case 3
            rec_pos_b = [3171 1419 1734 1139];
            labels_str = {'GT1'};
            image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_b,...
                        labels_str, 'Font', 'LucidaTypewriterBold', 'FontSize',42, ...
                        'TextColor','white', 'Color', 'blue', 'TextBoxOpacity',0.7,'LineWidth',12);            
            rec_pos_g = [3411 2631 642 835];
            labels_str = {'GT2'};
            image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_g,...
                        labels_str, 'Font', 'LucidaTypewriterBold', 'FontSize',42, ...
                        'TextColor','black', 'Color', 'green', 'TextBoxOpacity',0.7,'LineWidth',12);
            rec_pos_r = [3578 3800 457 366];
            labels_str = {'GT3'};
            image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_r,...
            labels_str, 'Font', 'LucidaTypewriterBold','FontSize',36,'TextColor','white',...
                        'Color', 'red', 'TextBoxOpacity',0.7,'LineWidth',6);     
        otherwise
            error('Unsupported ROI');
    end
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
    
    axis on, grid on;
    %title('Ground truth overalyed on Kalyan cropped image');
    colormap(map);
    colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
    switch r
        case 1
            axis([3000 ncols 3900 nrows]);
        case 3
            axis([3100 5000 1300 4200]);
        otherwise
            error('Unsupported ROI');
    end
end
