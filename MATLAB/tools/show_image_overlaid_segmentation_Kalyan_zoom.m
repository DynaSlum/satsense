% show_image_overlaid_segmentation - generating a figure with overlaid
% segmentation on it


%% params
[ paths, processing_params, exec_flags] = config_params_Kalyan();

[data_dir, masks_dir, ~, ~, ~, segmentation_dir] = v2struct(paths);
[vocabulary_size, best_tile_size, best_tile_size_m, ~, ~, ~, roi] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

fs = 18;

str = ['px' num2str(best_tile_size) 'm' num2str(best_tile_size_m)];

vis_initial_result = false;
vis_filled_result = false;
vis_final_result = true;

overlay = true;
result_only = false;
mapw = [0 0 1; 0 1 0; 1 0 0; 1 1 1]; % Blue, Green, Red, White = 1,2,3, NaN
map = [0 0 1; 0 1 0; 1 0 0]; % Blue, Green, Red = 1,2,3

%% display
image_fname = ['Mumbai_P4_R1C1_3_ROI_clipped.tif'];
image_fullfname = fullfile(data_dir, image_fname);
segm_fname = fullfile(segmentation_dir, ['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '_' roi '.mat']);

%% load the data and the result
image_data = imread(image_fullfname);
[nrows, ncols, ~] = size(image_data);
load(segm_fname); % contains segmented_image (every 10th pixel); filled_segm_image and segmented_image_denoised


%% visualize initial result
if vis_initial_result
    if result_only
        RGB1 = ind2rgb(segmented_image, mapw);
        figure; imshow(RGB1, mapw); %title('Segmented Bangalore cropped image');
        %xlabel('Every 10th pixel is processed');
        %colorbar('Ticks', [0.1 0.35 0.65 0.9], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum', 'Not Processed'});
        handleToColorBar = colorbar('Ticks', [0.1 0.35 0.65 0.9]);
        set(handleToColorBar,'YTickLabel', []);
        hYLabel = ylabel(handleToColorBar,['BuiltUp       NonBuiltUp      Slum      Unprocessed']);
        set(hYLabel,'Rotation',90);
        set(hYLabel,'FontSize',fs);
        %axis on, grid on
        
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
        % axis on, grid on;
        % title('Segmentation overlaid on Kalyan cropped image');
        colormap(mapw);
        handleToColorBar = colorbar('Ticks', [0.1 0.35 0.65 0.9]);
        set(handleToColorBar,'YTickLabel', []);
        hYLabel = ylabel(handleToColorBar,['BuiltUp       NonBuiltUp      Slum      Unprocessed']);
        set(hYLabel,'Rotation',90);
        set(hYLabel,'FontSize',fs);
        
    end
end

%% filled result
if vis_filled_result
    if result_only
        RGB2 = ind2rgb(filled_segm_image, map);
        figure; imshow(RGB2, map); %title('Filled segmented Kalyan cropped image');
        %xlabel('Missing pixels filled with majority vote fom a 20x20 window');
        % colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
        handleToColorBar = colorbar('Ticks', [0.2 0.5 0.8]);
        set(handleToColorBar,'YTickLabel', []);
        hYLabel = ylabel(handleToColorBar,['BuiltUp        NonBuiltUp       Slum']);
        set(hYLabel,'Rotation',90);
        set(hYLabel,'FontSize',fs);
        %  axis on, grid on
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
        % axis on, grid on;
        % title('Filled segmentation overlaid on Kalyan cropped image');
        colormap(map);
        %colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
        handleToColorBar = colorbar('Ticks', [0.2 0.5 0.8]);
        set(handleToColorBar,'YTickLabel', []);
        hYLabel = ylabel(handleToColorBar,['BuiltUp        NonBuiltUp       Slum']);
        set(hYLabel,'Rotation',90);
        set(hYLabel,'FontSize',fs);
        
    end
end

%% final result
if vis_final_result
    if result_only
        RGB3 = ind2rgb(segmented_image_denoised, map);
        figure; imshow(RGB3, map); %title('Denoised segmented Kalyan cropped image');
        %xlabel('Majority filter of size 40x40 is used');
        % colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
        handleToColorBar = colorbar('Ticks', [0.2 0.5 0.8]);
        set(handleToColorBar,'YTickLabel', []);
        hYLabel = ylabel(handleToColorBar,['BuiltUp        NonBuiltUp       Slum']);
        set(hYLabel,'Rotation',90);
        set(hYLabel,'FontSize',fs);
        % axis on, grid on
        
    end
    if overlay
        %% add special rectangles of interest
        rec_pos_g = [1055 2500 145 180];
        labels_str = {'S3'};
        image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_g,...
            labels_str, 'Font', 'LucidaTypewriterBold', 'FontSize',42, ...
            'TextColor','black', 'Color', 'green', 'TextBoxOpacity',0.7,'LineWidth',12);
        
        rec_pos_b = [750 3350 200 150; 1020 3555 150 105];
        labels_str = {'S1';'S2'};
        image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_b,...
            labels_str, 'Font', 'LucidaTypewriterBold','FontSize',36,'TextColor','white',...
            'Color', 'blue', 'TextBoxOpacity',0.7,'LineWidth',6);
        
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
        % axis on, grid on;
        % title('Denoised segmentation overlaid on Kalyan cropped image');
        colormap(map);
        %  colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
        handleToColorBar = colorbar('Ticks', [0.2 0.5 0.8]);
        set(handleToColorBar,'YTickLabel', []);
        
        hYLabel = ylabel(handleToColorBar,['BuiltUp              NonBuiltUp            Slum']);
        
        %   hYLabel = ylabel(handleToColorBar,['BuiltUp                             NonBuiltUp                           Slum']);
        
        
        set(hYLabel,'FontSize',fs);
        set(hYLabel,'FontSize',fs);
        
    end
end
%% zoom
axis([100 1500 2400 3800 ]);
%axis on, grid on;