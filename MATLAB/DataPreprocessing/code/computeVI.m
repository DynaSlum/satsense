function [ VI] = computeVI( rgbI, method, visualize)
%% generates a vegetation index (VI) from the RGB imput image by some VI method
% rgbI - the rgb image
% method - the vegetation index method. Possible options:
%   RG - red-green
%   RB - red-blue
%   VVI - visible vegetation index
%   TGI - triangular greenness indes
%   VDVI - visible band-difference vegetation index
%   VARI - Visible Atmospherically Resistant Index
%   HSVDT - conversion to HSV  and using decision tree
% visualize - optional visualization flag; default is true
%
% VI - the vegetation index values matrix
% For Testing use test_computeVI

if nargin < 3
    visualize = true;
end
if nargin < 2
    error('computeVI requires at least 2 arguments - an RGB image and a vegetation index method!');
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
        VI = (1-abs((red-r0)./(red+r0))).*(1-abs((green-g0)./ ...
                (green + g0))).*(1-abs((blue-b0)./(blue+b0)))*255;
    case 'TGI'
        VI = green - 0.35*red -0.61*blue;   
    case 'VDVI'
        VI = (2*green - red - blue)./(2*green + red + blue);
    case 'VARI'
        VI = (green - red)./(green + red - blue)*255;  
    case 'RB'
        VI = (red - blue)./(red + blue);
    case 'RG'
        VI = (red - green)./(red + green);
    case 'HSVDT'
        %Set the hue value to zero if it isless than 50 or great than 150
        H((H < 50) | (H > 150)) = 0;
        VI = H;
end
        

%% visualize
if visualize
    figure;
    
    subplot(121); image(rgbI); axis image; axis on, grid on, title('RGB Image');
    subplot(122); 
    switch method
        case 'HSVDT'
            image(VI); title('Vegetation index: HSVDT');   
        case 'VVI'
            image(VI); title('Vegetation index: VVI');    
        case 'TGI'
            image(VI); title('Vegetation index: TGI');
        case 'VDVI'
            image(VI*255); title('Vegetation index: VDVI (*255)');
        case 'VARI'
            image(VI); title('Vegetation index: VARI');            
        case 'RB'
            image(VI*255); title('Vegetation index: RB (*255)');
        case 'RG'
            image(VI*255); title('Vegetation index: RG (*255)');
    end
   
    axis image; axis on, grid on, colormap(jet); colorbar('eastoutside');
end

