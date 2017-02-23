% Testing of classifyEachPixelFromTile

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

n = 1;
tile_sizes = [100];
tile_sizes_m = [80];
vocabulary_size = [50];
tile_size = tile_sizes(n);
tile_size_m = tile_sizes_m(n);
%stepY = 1;
stepY = 5;
stepX = stepY;
tile_step = [stepX stepY];

data_path = fullfile(root_dir, 'Data','Kalyan','Rasterized_Lourens');
sav_path = fullfile(root_dir, 'Results', 'Segmentation');
image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';
str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
sav_path_classifier = fullfile(root_dir, 'Results','Classification3Classes','Classifiers');
fname = fullfile(sav_path_classifier, ['trained_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '.mat']) ;
load(fname); % contains categoryClassifier

image_fullfname = fullfile(data_path, image_fname);

%% classify each pixel
tic
[ segmented_image] = classifyEachPixelFromTile( image_fullfname, ...
    [tile_size tile_size], tile_step, categoryClassifier);
disp('Done!');
toc
%% visualize
map = [0 0 1; 0 1 0; 1 0 0];
RGB = ind2rgb(segmented_image,map);
figure; imshow(RGB); title('Segmented Kalyan cropped image');
legend({'BuiltUp'; 'NonBuiltUp'; 'Slum'});
axis on, grid on

%% save
sav_fname = fullfile(sav_path,['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '.mat']);
save(sav_fname,'segmented_image');

disp('DONE!');
