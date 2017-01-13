function [feature_table] = createFeatureTable( train_imds, feature_vectors)

%% createFeatureTable   make table from the feature vectors for the classifier 
%   INPUT:
%   train_imds -the train datastore 
%   feature_vectors - the histograms of visual word occurrences, specified as 
%                   M-by-bag as output by createVisualVocabulary
%   OUTPUT:
%   feature_table - the feature vecors as table with column classes for the
%                   class labels 
% For Testing use test_createFeatureTable
% Note: see also https://nl.mathworks.com/help/vision/ug/...
% image-classification-with-bag-of-visual-words.html
% and 
% https://nl.mathworks.com/matlabcentral/fileexchange/...
% 58320-demos-from--object-recognition--deep-learning--webinar/content/...
% DeepLearningWebinar/Demo1_BagOfFeatures/Scene_Identification.m

%% convert to table
feature_table= array2table(feature_vectors);

%% add class labels
feature_table.class_label = train_imds.Labels;
