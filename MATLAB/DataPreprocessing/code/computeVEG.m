function [ VEG, vegetation_mask_raw] = computeVEG( rgbI, method, thresh, visualize)
%% generates a vegetation mask from the RGB imput image using some Vegetation Index (VEG)
% rgbI - the rgb image
% method - the vegetation index method. Possible options:
%   RG - red-green
%   RB - red-blue
%   VVI - visible vegetation index
%   TGI - triangular greenness indes
%   VDVI - visible band-difference vegetation index
%   VARI - Visible Atmospherically Resistant Index
%   HSVDT - conversion to HSV  and using decision tree
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
    case 'HSVDT'
        hsv= rgb2hsv(rgbI);
        H = hsv(:, :, 1).*255;
        S = hsv(:, :, 2).*255;
        V = hsv(:, :, 3).*255;
end

%% compute VEG and mask
switch method
    case 'VVI'
        VEG = (1-abs((red-r0)./(red+r0))).*(1-abs((green-g0)./(green + g0))).*(1-abs((blue-b0)./(blue+b0)))*255;
        maxVEG = max(max(VEG));
        %minVEG = min(min(VEG));
        thresh = 25;
    case 'TGI'
        VEG = green - 0.35*red -0.61*blue;   
    case 'VDVI'
        VEG = (2*green - red - blue)./(2*green + red + blue);
    case 'VARI'
        VEG = (green - red)./(green + red - blue)*255;  
        thresh1 = 0; thresh2 = 50;
    case 'RB'
        VEG = (red - blue)./(red + blue);
    case 'RG'
        VEG = (red - green)./(red + green);
        %thresh =10/255;
    case 'HSVDT'
        %Set the hue value to zero if it isless than 50 or great than 150
        H((H < 50) | (H > 150)) = 0;
        %H(H > 49 & H < 60 & S > 5 & S < 50 & V > 150) = 0;
        %Thresholding
        T = 49; %T can be any value in [1, 49]
        thresh = T./255;
        VEG = H;
end
        
if strcmp(method,'VARI')
    vegetation_mask_raw = logical((VEG > thresh1)&(VEG < thresh2));
elseif strcmp(method, 'RG') || strcmp(method, 'RB') %|| strcmp(method,'TGI')
    vegetation_mask_raw = logical(VEG < thresh);
else
    vegetation_mask_raw = logical(VEG > thresh);
end


%% filter the vegetation mask
se = strel('square',20);
vegetation_mask_remove_noise2000 = bwareaopen(vegetation_mask_raw, 2000);
vegetation_mask_open = imopen(vegetation_mask_remove_noise2000,se);
vegetation_mask_close = imclose(vegetation_mask_open,se);
vegetation_mask_remove_noise1000 = bwareaopen(vegetation_mask_close, 1000);
vegetation_mask_fill = imfill(vegetation_mask_remove_noise1000,'holes');
vegetation_mask =vegetation_mask_fill;

%% visualize
if visualize
    figure;
    
    subplot(221); imshow(rgbI); axis on, grid on, title('RGB Image');
    subplot(223); imshow(vegetation_mask_raw); axis on, grid on, title('Raw vegetation mask');    
    subplot(224); imshow(vegetation_mask); axis on, grid on, title('Vegetation mask');    
    switch method
        case 'HSVDT'
            subplot(222); image(uint8(VEG)); 
            axis image; axis on, grid on, colormap(jet); colorbar;
            title('Vegetation index: HSVDT');   
        case 'VVI'
            subplot(222); image(uint8(VEG)); 
            axis image; axis on, grid on, colormap(jet); colorbar;
            title('Vegetation index: VVI');    
        case 'TGI'
            subplot(222); image(VEG); 
            axis image; axis on, grid on, colormap(jet); colorbar;
            title('Vegetation index: TGI');
        case 'VDVI'
            subplot(222); imagesc(VEG*255); 
            axis image; axis on, grid on, colormap(jet); colorbar;
            title('Vegetation index: VDVI');
        case 'VARI'
            subplot(222); image(VEG); 
            axis image; axis on, grid on, colormap(jet); colorbar;
            title('Vegetation index: VARI');            
        case 'RB'
            subplot(222); image(VEG*255); 
            axis image; axis on, grid on, colormap(jet); colorbar;
            title('Vegetation index: RB');
        case 'RG'
            subplot(222); image(VEG*255); 
            axis image; axis on, grid on, colormap(jet); colorbar;            
            title('Vegetation index: RG');
    end
end

