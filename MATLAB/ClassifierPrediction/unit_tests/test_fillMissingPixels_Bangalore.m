% Testing of fillMissingPixels

%% parameters
[ paths, processing_params, exec_flags] = config_params_Bangalore();

[~, ~, ~, segmentation_dir] = v2struct(paths);
[vocabulary_size, best_tile_size, best_tile_size_m, ~, window_size, ROIs] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);


str = ['px' num2str(best_tile_size) 'm' num2str(best_tile_size_m)];
num_ROIs = length(ROIs);

%% fill missing pixels
for r = 1:num_ROIs
    roi = ROIs{r};
    
    if verbose
        disp(['Processing ROI: ', roi, '...']);
    end
    
    inp_fname = fullfile(segmentation_dir,['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '_' roi '.mat']);
    load(inp_fname); % contains segmented_image
    
    %% fill missing pixels
    tic
    [ filled_segm_image] = fillMissingPixels( segmented_image, window_size);
    disp('Done!');
    toc
    %% visualize
    if visualize
        
        map = [0 0 1; 0 1 0; 1 0 0;]; % Blue, Green, Red = 1,2,3
        RGB = ind2rgb(filled_segm_image,map);
        figure; imshow(RGB, map); title('Segmented Bangalore cropped ROI');
        xlabel(['Misssing pixles filled with majority vote from a window: ',...
            num2str(window_size(1)), ' x ', num2str(window_size(2))] );
        colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', ...
            {'BuiltUp', 'NonBuiltUp', 'Slum'});
        axis on, grid on
    end
    %% save
    if sav
        save(inp_fname,'filled_segm_image','-append');
    end
    disp('DONE!');
    
end % for ROI

disp('DONE!!!');