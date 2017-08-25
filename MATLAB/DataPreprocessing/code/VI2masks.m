function [ urbanMask, vegetationMask] = VI2masks( VI, visualize)
%% generates a urban mask froom vegetation index (VI)
% VI - the vegetation index values matrix
% visualize - optional visualization flag; default is true
%
% For Testing use test_VI2urban

if nargin < 2
    visualize = true;
end
if nargin < 1
    error('VI2urban requires at least 1 argument - a vegetation index matrix!');
end

%% threshold the VI matrix with it's mean value
thresh = mean(mean(VI));

bw = logical(VI <= thresh);

%% parameters
% [w,h] = size(VI);
% d = sqrt(w^2 + h^2);
% A = w*h;
areaNoise = 2000;
sizeSE = 20;
%% post-processing
% nose removal
bwn = ~bwareaopen(~bw, areaNoise/2);
SE =  strel('square', sizeSE); % for removing shadows
bwc = imclose(bwn, SE);
urbanMask = ~bwareaopen(~bwc, areaNoise);
bwv = ~urbanMask;
bwvn = bwareaopen(bwv, areaNoise);
bwvnc = imclose(bwvn, SE);
vegetationMask = bwareaopen(bwvnc, 2*areaNoise);
%% visualize
if visualize
    figure;
    
    subplot(231); image(VI); axis image; axis on, grid on, title('Vegetation index: VVI');
    subplot(232); imshow(bw); axis image; axis on, grid on, title('Thresholded (mean) VVI');
    subplot(233); imshow(bwn); axis image; axis on, grid on, title('Area Opening- noise removal');
    subplot(234); imshow(bwc); axis image; axis on, grid on, title('Closing');
    subplot(235); imshow(urbanMask); axis image; axis on, grid on, title('UrbanMask');
    subplot(236); imshow(vegetationMask); axis image; axis on, grid on, title('VegetationMask');
end
