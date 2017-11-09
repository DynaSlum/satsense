% Testing of fillMissingPixels for Bangalore ROIs

%% parameters
[ paths, processing_params, exec_flags] = config_params_Bangalore();

[~, ~, ~, segmentation_dir] = v2struct(paths);
[vocabulary_size, best_tile_size, best_tile_size_m, ~, ~, window_size, ROIs] = ...
    v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);
disp('Window_size: '); disp(window_size);

str = ['px' num2str(best_tile_size) 'm' num2str(best_tile_size_m)];
num_ROIs = length(ROIs);

%% filter  missing pixels
for r = 1:num_ROIs
    roi = ROIs{r};
    
    if verbose
        disp(['Processing ROI: ', roi, '...']);
    end
    
    inp_fname = fullfile(segmentation_dir,['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '_' roi '.mat']);
    load(inp_fname); % contains segmented_image
    tic
    [ segmented_image_denoised] = majorityFilterSegmentation( filled_segm_image, window_size);
    disp('Done!');
    toc
    %% visualize
    map = [0 0 1; 0 1 0; 1 0 0 ]; % Blue, Green, Red = 1,2,3
    RGB = ind2rgb(segmented_image_denoised,map);
    figure; imshow(RGB, map); title('Denoised segmented Bangalore ROI');
    xlabel(['Majority filter of size: ',...
        num2str(window_size(1)), ' x ', num2str(window_size(2)) ' is used'] );    
    colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', ...
        {'BuiltUp', 'NonBuiltUp', 'Slum'});
    axis on, grid on
    
    %% save
    if sav
        save(inp_fname,'segmented_image_denoised','-append');
    end;
    
    disp('DONE!');
end % for ROI

disp('DONE!!!');