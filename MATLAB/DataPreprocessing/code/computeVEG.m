function [ VEG, vegetation_mask] = computeVEG( rgbI, visualize)
%% generates a vegetation mask from the RGB imput image using some Vegetation Index (VEG)
% rgbI - the rgb image
% visualize - optional visualization flag; default is true
%
% VEG - the vegetaiton index values
% vegetation_mask - binary mask where VEG >0
% For Testing use test_computeVEG

if nargin < 2
    visualize = true;
end
if nargin < 1
    error('computeVEG requires at least 1 argument - an RGB image!');
end

%% some parameters and reference values
rgb0 = [30 50 0]+10;
r0 = rgb0(1);
g0 = rgb0(2);
b0 = rgb0(3);

%% get the color channels
red = double(rgbI(:,:,1));
green = double(rgbI(:,:,2));
blue = double(rgbI(:,:,3));

red = red + 10;
green = green + 10;
blue = blue + 10;

% % normalized color channels
% color_sum = red + green + blue;
% norm_red = red./color_sum;
% norm_green = green./color_sum;
% norm_blue = blue./color_sum;


%% compute VVI and mask
VEG = (1-abs((red-r0)./(red+r0))).*(1-abs((green-g0)./(green + g0))).*(1-abs((blue-b0)./(blue+b0)))*255;
maxVEG = max(max(VEG))
%minVEG = min(min(VEG))
thresh = (maxVEG *0.2)
vegetation_mask = logical(VEG > thresh);
%% visualize 10% of the points

if visualize
    figure;
    
    subplot(231); imshow(rgbI); axis on, grid on, title('RGB Image');
    subplot(232); imagesc(red); axis image;axis on, grid on, title('Red channel');
    subplot(233); imagesc(green);axis image; axis on, grid on, title('Green channel');
    subplot(234); imshow(vegetation_mask); axis on, grid on, title('Vegetation mask');
    subplot(235); imagesc(uint8(VEG)); axis image; axis on, grid on, title('VEG');colormap(gray(255)); colorbar;
end

