function [ VEG, vegetation_mask] = computeVEG( rgbI, method, thresh, visualize)
%% generates a vegetation mask from the RGB imput image using some Vegetation Index (VEG)
% rgbI - the rgb image
% method - the vegetation index method. Possible options:
%   RG - red-green
%   RB - red-blue
%   VVI - visible vegetation index
% thresh - threshold for obtaining the vegetation mask from the VEG,
%          vegetation index values matrix
% visualize - optional visualization flag; default is true
%
% VEG - the vegetation index values matrix
% vegetation_mask - binary mask where VEG >0
% For Testing use test_computeVEG

if nargin < 4
    visualize = true;
end
if nargin < 3 || isempty(thresh)
    thresh = 0;
end
if nargin < 2
    error('computeVEG requires at least 2 arguments - an RGB image and a vegetation index method!');
end


%% get the color channels
red = double(rgbI(:,:,1));
green = double(rgbI(:,:,2));
blue = double(rgbI(:,:,3));

switch method
    case 'VVI'
        rgb0 = [30 50 0]+10;
        r0 = rgb0(1);
        g0 = rgb0(2);
        b0 = rgb0(3);
        
        red = red + 10;
        green = green + 10;
        blue = blue + 10;
end

%% compute VEG and mask
switch method
    case 'VVI'
        VEG = (1-abs((red-r0)./(red+r0))).*(1-abs((green-g0)./(green + g0))).*(1-abs((blue-b0)./(blue+b0)))*255;
        maxVEG = max(max(VEG));
        %minVEG = min(min(VEG));
        thresh = (maxVEG *0.2);
        
    case 'RB'
        VEG = (red - blue)./(red + blue);
    case 'RG'
        VEG = (red - green)./(red + green);
end
        
vegetation_mask = logical(VEG > thresh);

%% visualize
if visualize
    figure;
    
    subplot(221); imshow(rgbI); axis on, grid on, title('RGB Image');
    %subplot(232); imagesc(red); axis image;axis on, grid on, title('Red channel');
    %subplot(233); imagesc(green);axis image; axis on, grid on, title('Green channel');
    subplot(223); imshow(vegetation_mask); axis on, grid on, title('Vegetation mask');    
    switch method
        case 'VVI'
            subplot(224); imagesc(uint8(VEG)); 
            axis image; axis on, grid on, colormap(gray(255)); colorbar;
            title('Vegetation index: VVI');            
        case 'RB'
            subplot(224); imagesc(uint8(VEG)*255); 
            axis image; axis on, grid on, colormap(gray(255)); colorbar;
            title('Vegetation index: RB');
        case 'RG'
            subplot(224); imagesc(uint8(VEG)*255); 
            axis image; axis on, grid on, colormap(gray(255)); colorbar;            
            title('Vegetation index: RG');
    end
end

