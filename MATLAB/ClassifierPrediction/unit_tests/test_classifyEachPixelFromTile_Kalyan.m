% Testing of classifyEachPixelFromTile_Kalyan

%% params
[ paths, processing_params, exec_flags] = config_params_Kalyan();

[data_dir, masks_dir, datastores_dir, classifier_dir, performance_dir, segmentation_dir] = v2struct(paths);
[vocabulary_size, best_tile_size, best_tile_size_m, tile_step, ~,~, roi] = ...
    v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);


str = ['px' num2str(best_tile_size) 'm' num2str(best_tile_size_m)];
fname = fullfile(classifier_dir, ['trained_SURF_SVM_Classifier_' num2str(vocabulary_size) '_' str '.mat']) ;
load(fname); % contains categoryClassifier


%% segmentation

image_fname = ['Mumbai_P4_R1C1_3_ROI_clipped.tif'];
image_fullfname = fullfile(data_dir, image_fname);



%% classify each pixel
tic
[ segmented_image] = classifyEachPixelFromTile( image_fullfname, ...
    [best_tile_size best_tile_size], tile_step, categoryClassifier);
disp('Done!');
toc
%% visualize
if visualize
    map = [0 0 1; 0 1 0; 1 0 0; 1 1 1]; % White, Blue, Green, Red, White = 1,2,3, NaN
    RGB = ind2rgb(segmented_image,map);
    
    figure; imshow(RGB, map); title('Segmented Kalyan cropped ROI (every 10th pixel)');
    %legend('Not processed','BuiltUp', 'NonBuiltUp', 'Slum');
    colorbar('Ticks', [0.1 0.35 0.65 0.9], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum', 'Not Processed'});
    axis on, grid on
end

%% save
if sav
    sav_fname = fullfile(segmentation_dir,['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '_' roi '.mat']);
    save(sav_fname,'segmented_image');
end
disp('DONE!');
