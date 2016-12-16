%% load the image data
cl;
data_path = 'C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\';
importfile(fullfile(data_path,'Mumbai_P4_R1C1_3_clipped_rgb.tif'));
image_data = Mumbai_P4_R1C1_3_clipped_rgb;
clear Mumbai_P4_R1C1_3_clipped_rgb

nrows = size(image_data,1);
ncols = size(image_data,2);

%% load the mask data
importfile(fullfile(data_path,'rough_builtup_mask.tif'));
importfile(fullfile(data_path,'all_slums.tif'));

slums_mask = all_slums;
clear all_slums


%% derive builup (w/o slums) and nonbuiltup masks
builtup_mask = false(nrows, ncols);
nonbuiltup_mask = false(nrows, ncols);

for r = 1:nrows
    for c = 1:ncols
        if and(rough_builtup_mask(r,c),not(slums_mask(r,c)))
            builtup_mask(r,c) = true;
        end
        if and(not(rough_builtup_mask(r,c)),not(slums_mask(r,c)))
            nonbuiltup_mask(r,c) = true;
        end
    end
end

%% save the derived masks
imwrite(builtup_mask,fullfile(data_path,'builtup_mask.tif'));
imwrite(nonbuiltup_mask,fullfile(data_path,'nonbuiltup_mask.tif'));

%% prepare colored overlays
red = cat(3, ones(nrows,ncols), zeros(nrows, ncols), zeros(nrows,ncols));
green = cat(3, zeros(nrows,ncols), ones(nrows,ncols), zeros(nrows,ncols));
blue = cat(3, zeros(nrows,ncols), zeros(nrows,ncols), ones(nrows,ncols));

%% display the image daya
figure;imshow(image_data);
axis on, grid on;
title('Image data');

%% display the image daya and overlap the overlays
figure; imshow(image_data);
hold on;
hr=imshow(red);
set(hr, 'AlphaData', 0.2*slums_mask);
hg=imshow(green);
set(hg, 'AlphaData', 0.2*nonbuiltup_mask);
hb=imshow(blue);
set(hb, 'AlphaData', 0.2*builtup_mask);
hold off
axis on, grid on;
title('Image data with overalyed labels: red- slums, blue - built-up(rough) and green - non-built-up (rough)');

%% display the image daya and overlap the non built-up overlay
figure; imshow(image_data);
hold on;
hb=imshow(green);
set(hb, 'AlphaData', 0.3*nonbuiltup_mask);
hold off
axis on, grid on;
title('Image data with overalyed rough non built-up areas');

%% display the image daya and overlap the built-up overlay
figure; imshow(image_data);
hold on;
hg=imshow(blue);
set(hg, 'AlphaData', 0.3*builtup_mask);
hold off
axis on, grid on;
title('Image data with overalyed rough built-up areas');

%% display the image daya and overlap the slum overlay
figure; imshow(image_data);
hold on;
hr=imshow(red);
set(hr, 'AlphaData', 0.3*slums_mask);
hold off
axis on, grid on;
title('Image data with overlayed slums');
