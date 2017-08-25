function [ ] = showMaskOverROI( rgbROI, mask, col)
%% visualizes mask over RGB ROI
% rgbROI- the RGB ROI
% mask - the binary mask to be displayed over the ROI

if nargin < 2
    error('showMaskOverROI requires at least 2 arguments - an rbg ROI and a mask!');
end

% compute mask boundaties
[B] = bwboundaries(mask);

%% visualize
figure;
image(rgbROI);
axis image, axis on; grid on;
title('Vegetation mask over ROI');
hold on;
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), col, 'LineWidth', 1);
end
hold off

