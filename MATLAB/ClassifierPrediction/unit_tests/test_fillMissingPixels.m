% Testing of fillMissingPixels

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
wsY = 20;wsX = wsY;
window_size = [wsX wsY];

sav_path = fullfile(root_dir, 'Results', 'Segmentation');
%image_fname = 'Mumbai_P4_R1C1_3_clipped_rgb.tif';
str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
inp_fname = fullfile(sav_path,['SegmentedImage_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '.mat']);
load(inp_fname); % contains segmented_image

%% fill missing pixels
tic
[ filled_segm_image] = fillMissingPixels( segmented_image, window_size);
disp('Done!');
toc
%% visualize
map = [0 0 1; 0 1 0; 1 0 0;]; % Blue, Green, Red = 1,2,3
RGB = ind2rgb(filled_segm_image,map);
figure; imshow(RGB, map); title('Segmented Kalyan cropped image');
xlabel(['Misssing pixles filled with majority vote from a window: ',...
    num2str(wsX), ' x ', num2str(wsY)] );
%legend('Not processed','BuiltUp', 'NonBuiltUp', 'Slum');
colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', ...
    {'BuiltUp', 'NonBuiltUp', 'Slum'});
axis on, grid on

%% save
save(inp_fname,'filled_segm_image','-append');

disp('DONE!');
