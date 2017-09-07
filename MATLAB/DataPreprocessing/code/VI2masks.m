function [urbanMask, vegetationMask] = VI2masks( VI, thresh_type, visualize)
%% generates a urban mask froom vegetation index (VI)
% VI - the vegetation index values matrix
% thresh_type - the type of thresholidng, can be 'mean' or 'fixed'
% visualize - optional visualization flag; default is true
%
% For Testing use test_VI2urban

if nargin < 3
    visualize = true;
end
if nargin < 2
    thresh_type = 'mean';
end
if nargin < 1
    error('VI2urban requires at least 1 argument - a vegetation index matrix!');
end

%% threshold the VI matrix with it's mean value
switch thresh_type
    case 'mean'    
        thresh = mean(mean(VI));
    case 'fixed'
        thresh = 25;
    otherwise
        error('Unknown threshold type! Chose `mean` or `fixed` ');
end

bw = logical(VI > thresh);

%% parameters
% [w,h] = size(VI);
% d = sqrt(w^2 + h^2);
% A = w*h;
areaNoise = 5000;
sizeSE = 20;
%% post-processing
% nose removal
% bwn = ~bwareaopen(~bw, areaNoise/2);
SE =  strel('square', sizeSE);
% denoise
bwn = bwareaopen(bw, areaNoise);
% closing
bwc = imclose(bwn, SE);
% denoise
bwcn = ~bwareaopen(~bwc, areaNoise);
% vegetation and urban
vegetationMask = bwcn;
urbanMask = ~vegetationMask;
%% visualize
if visualize
    figure;
    
    subplot(231); image(VI); colormap(jet); axis image; axis on, grid on, title('Vegetation index: VVI');
    subplot(232); imshow(bw); axis image; axis on, grid on, title('Thresholded (mean) VVI');
    subplot(233); imshow(bwn); axis image; axis on, grid on, title('Area Opening- noise removal');
    subplot(234); imshow(bwc); axis image; axis on, grid on, title('Closing');
    subplot(236); imshow(urbanMask); axis image; axis on, grid on, title('UrbanMask');
    subplot(235); imshow(vegetationMask); axis image; axis on, grid on, title('VegetationMask (denoised)');
end
