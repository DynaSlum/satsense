function [ num_points] = getSURFfeatures( rgbI, class_label, visualize)
%% countss the valid SURF points of an image of a given class and optionally visualizes 
% rgbI - the rgb image
% class label - the class label string
% visualize - optional visualizatin glag; default is true
%
% num_points -the number of valid SURF points
% For Testing use test_getSURFfeatures

if nargin < 3
    visualize = true;
end
if nargin < 2
    error('getSURFfeatures requires at least 2 arguments');
end

%% prepare image
I = rgb2gray(rgbI);

%% compute SURF points and descriptor/features
points = detectSURFFeatures(I);
[~, valid_points] = extractFeatures(I, points);
num_points = valid_points.Count;
%% visualize 10% of the points

if visualize
    num_display_points = fix(0.1*num_points);
    
    figure; imshow(rgbI); axis on; hold on;
    plot(valid_points.selectStrongest(num_display_points),...
        'showOrientation',true);
    
    title([' 10% of the ', num2str(num_points),...
        ' SURF points for first image of class ' class_label], ...
        'Interpreter', 'None');
end
