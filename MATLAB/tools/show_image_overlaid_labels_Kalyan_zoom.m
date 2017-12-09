%% params
[ paths, processing_params, exec_flags] = config_params_Kalyan();

[data_dir, masks_dir, ~, ~] = v2struct(paths);
[~, ~, ~, ~, ~, ~, roi] = v2struct(processing_params);
[verbose, visualize, sav] = v2struct(exec_flags);

fs = 18;
%% displaying

image_data = imread(fullfile(data_dir,['Mumbai_P4_R1C1_3_ROI_clipped.tif']));

nrows = size(image_data,1);
ncols = size(image_data,2);

%% load the mask data
slum_mask = imread(fullfile(masks_dir,['Kalyan_' roi '_slumMask.tif']));
slum_mask = slum_mask * 255;
builtup_mask = imread(fullfile(masks_dir,['Kalyan_' roi '_urbanMask.tif']));
% builtup_mask = builtup_mask * 255;
nonbuiltup_mask = imread(fullfile(masks_dir,['Kalyan_' roi '_vegetationMask.tif']));
% nonbuiltup_mask = nonbuiltup_mask * 255;
%% prepare colored overlays
red = cat(3, ones(nrows,ncols), zeros(nrows, ncols), zeros(nrows,ncols));
green = cat(3, zeros(nrows,ncols), ones(nrows,ncols), zeros(nrows,ncols));
blue = cat(3, zeros(nrows,ncols), zeros(nrows,ncols), ones(nrows,ncols));

% %% display the image daya
% figure;imshow(image_data);
% axis on, grid on;
% title('Image data');

% add special rectangles of interest

rec_pos_g = [120 2480 200 400];
labels_str = {'GT1'};
image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_g,...
    labels_str, 'Font', 'LucidaTypewriterBold', 'FontSize',42, ...
    'TextColor','black', 'Color', 'green', 'TextBoxOpacity',0.7,'LineWidth',12);
 rec_pos_r = [400 3100 280 180];
 labels_str = {'GT2'};
image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_r,...
    labels_str, 'Font', 'LucidaTypewriterBold','FontSize',36,'TextColor','white',...
    'Color', 'red', 'TextBoxOpacity',0.7,'LineWidth',6);

 rec_pos_b = [1320 2850 130 100];
 labels_str = {'GT3'};
image_data = insertObjectAnnotation(image_data,'Rectangle',rec_pos_b,...
    labels_str, 'Font', 'LucidaTypewriterBold','FontSize',36,'TextColor','white',...
    'Color', 'blue', 'TextBoxOpacity',0.7,'LineWidth',6);
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

% axis on, grid on;
%title('Ground truth overalyed on Kalyan cropped image');
colormap(map);
%colorbar('Ticks', [0.2 0.5 0.8], 'TickLabels', {'BuiltUp', 'NonBuiltUp', 'Slum'});
handleToColorBar = colorbar('Ticks', [0.2 0.5 0.8]);
set(handleToColorBar,'YTickLabel', []);

hYLabel = ylabel(handleToColorBar,['BuiltUp              NonBuiltUp            Slum']);

%        hYLabel = ylabel(handleToColorBar,['BuiltUp                             NonBuiltUp                           Slum']);

set(hYLabel,'Rotation',90);
set(hYLabel,'FontSize',fs);

%% zoom
axis([100 1500 2400 3800 ]);
%axis on, grid on;

