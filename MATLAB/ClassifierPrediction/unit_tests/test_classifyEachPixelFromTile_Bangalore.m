% Testing of classifyEachPixelFromTile_Bangalore

%% setup parameters

if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end
data_root_dir = fullfile(root_dir, 'Data','Bangalore','GEImages');
results_dir = fullfile(root_dir, 'Results','Bangalore');
if not(exist(results_dir,'dir')==7)
    mkdir(results_dir);
end

classifier_dir = fullfile(results_dir, 'Classification3Classes', 'Classifiers');
segmentation_dir = fullfile(results_dir, 'Segmentaiton');
    


data_dir = fullfile(data_root_dir, 'Clipped','fixed');
masks_dir = fullfile(data_root_dir, 'masks');

best_tile_size = [268];
best_tile_size_m = [40];
str = ['px' num2str(best_tile_size) 'm' num2str(best_tile_size_m)];

stepY = 20;
stepX = stepY;
tile_step = [stepX stepY];

ROIs = {
    'ROI1'
%     'ROI2'
%     'ROI3'
%     'ROI4'
%     'ROI5'
     };
num_ROIs = length(ROIs);


%% feature parameters
vocabulary_size = [50];
fname = fullfile(classifier_dir, ['trained_SURF_SVM_Classifier_' num2str(vocabulary_size) '_' str '.mat']) ;
load(fname); % contains categoryClassifier


%% execution flags
verbose = true;
visualize = true;
sav = true;


%% segmentation
for r = 1:num_ROIs
    roi = ROIs{r};
    
    if verbose
        disp(['Processing ROI: ', roi, '...']);
    end
    image_fname = ['Bangalore_' roi '.tif'];
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
        
        figure; imshow(RGB, map); title('Segmented Kalyan cropped image (every 5th pixel)');
        %legend('Not processed','BuiltUp', 'NonBuiltUp', 'Slum');
        colorbar('Ticks', [0.1 0.35 0.65 0.9], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum', 'Not Processed'});
        axis on, grid on
    end
    
    %% save
    if sav
        sav_fname = fullfile(segmentation_dir,['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '.mat']);
        save(sav_fname,'segmented_image');
    end
    disp('DONE!');
    
end % for ROI

disp('DONE!!!');