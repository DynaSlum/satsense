%% load the image data
cl;
data_path = 'C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\';
importfile(fullfile(data_path,'Mumbai_P4_R1C1_3_clipped.tif'));
image_data = Mumbai_P4_R1C1_3_clipped;
clear Mumbai_P4_R1C1_3_clipped

nrows = size(image_data,1);
ncols = size(image_data,2);

%% load the mask data
importfile(fullfile(data_path,'rough_urban_mask.tif'));
importfile(fullfile(data_path,'slums_municipality_raster_mask_8.tif'));

slums_mask = logical(slums_municipality_raster_mask_8);
clear slums_municipality_raster_mask_8


%% derive urban (w/o slums) and rural masks
urban_mask = false(nrows, ncols);
rural_mask = false(nrows, ncols);

for r = 1:nrows
    for c = 1:ncols
        if and(rough_urban_mask(r,c),not(slums_mask(r,c)))
            urban_mask(r,c) = true;
        end
        if and(not(rough_urban_mask(r,c)),not(slums_mask(r,c)))
            rural_mask(r,c) = true;
        end
    end
end

%% save the derived masks
imwrite(urban_mask,fullfile(data_path,'urban_mask.tif'));
imwrite(rural_mask,fullfile(data_path,'rural_mask.tif'));

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
set(hg, 'AlphaData', 0.2*urban_mask);
hb=imshow(blue);
set(hb, 'AlphaData', 0.2*rural_mask);
hold off
axis on, grid on;
title('Image data with overalyed labels: red- slums, green - urban(rough) and blue - rural (rough)');

%% display the image daya and overlap the rural overlay
figure; imshow(image_data);
hold on;
hb=imshow(blue);
set(hb, 'AlphaData', 0.3*rural_mask);
hold off
axis on, grid on;
title('Image data with overalyed rough rural areas');

%% display the image daya and overlap the urban overlay
figure; imshow(image_data);
hold on;
hg=imshow(green);
set(hg, 'AlphaData', 0.3*urban_mask);
hold off
axis on, grid on;
title('Image data with overalyed rough urban areas');

%% display the image daya and overlap the slum overlay
figure; imshow(image_data);
hold on;
hr=imshow(red);
set(hr, 'AlphaData', 0.3*slums_mask);
hold off
axis on, grid on;
title('Image data with overlayed slums');
