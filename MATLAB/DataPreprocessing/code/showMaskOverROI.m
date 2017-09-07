function [ ] = showMaskOverROI( rgbROI, mask, thresh_type, col)
%% visualizes mask over RGB ROI
% rgbROI- the RGB ROI
% thresh_type - the type of thresholidng, can be 'mean' or 'fixed'
% mask - the binary mask to be displayed over the ROI

if nargin < 4
    col = 'g';
end
if nargin < 3
    error('showMaskOverROI requires at least 3 arguments - an rbg ROI, a mask!');
end

% compute mask boundaties
[B] = bwboundaries(mask);

%% visualize
figure;
subplot(131); image(rgbROI); axis image; axis on, grid on, title('RGB Image ROI');
subplot(132); imshow(mask); axis image; axis on, grid on;
title(['Vegetation mask: ', thresh_type]);
subplot(133); image(rgbROI);
axis image, axis on; grid on;
title('Mask over ROI');
hold on;
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), col, 'LineWidth', 1);
end
hold off

