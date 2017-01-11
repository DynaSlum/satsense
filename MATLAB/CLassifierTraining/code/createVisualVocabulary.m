function [bagVW, feature_vectors] = createVisualVocabulary( train_imds,...
   vocabulary_size, strongest_features, point_selection, verbose, visualize)

%% createVisualVocabulary  wrapper aroind bagOfFeatures followed by encode with 
%   some preset parameters
%   INPUT:
%   train_imds -the train datastore 
%   vocabulary_size -the 'VocabularySize' parameter of bagOfFeatures.
%                   It is an integer scalar in the range [2, inf]. 
%                   The 'VocabularySize' value corresponds to K in the K-means 
%                   clustering algorithm used to quantize features into the
%                   visual vocabulary. Default value for 'VocabularySize'
%                   is 500, but 200 for vocabulary_size.
%   strongest_feature- the 'StrongestFeature' parameter of bagOfFeatures.
%                      Fraction of strongest features, a value in the range [0,1]. 
%                      Default value for 'StrongestFeatures' is 0.8, but
%                      for stringest_features is 0.6
%   point_selection- the 'PointSelection' parameter of bagOfFeatures.
%                     The default 'PointSelection' is 'Grid', but the default 
%                     of point_selection is 'Detector': the feature points 
%                     are selected using a speeded up robust feature (SURF) detector. 
%   verbose- the 'Verbose'. parameter of bagOfFeatures. The default is "true"
%   visualize- flag for fisualization of the feature vector. Default is "true"


%   OUPUT:
%   bagVW - the bag of VisualWords object returned by bagOfFeatures
%   feature_vectors- histograms of visual word occurrences, specified as 
%                   M-by-bag.VocabularySize vector, where M is the total 
%                   number of images in train_imds
%   
% For Testing use test_createVisualVocabulary
% Note: see also https://nl.mathworks.com/help/vision/ug/...
% image-classification-with-bag-of-visual-words.html
% and 
% https://nl.mathworks.com/matlabcentral/fileexchange/...
% 58320-demos-from--object-recognition--deep-learning--webinar/content/...
% DeepLearningWebinar/Demo1_BagOfFeatures/Scene_Identification.m

%% set up default parameters
if isempty(vocabulary_size)
    vocabulary_size = 200;
end
if isempty(strongest_features)
    strongest_features = 0.6;
end
if isempty(point_selection)
    point_selection = 'Detector';
end
if isempty(visualize)
    visualize = true;
end

if verbose
    tic
end
%% create bag of VW
bagVW = bagOfFeatures(train_imds, 'VocabularySize',vocabulary_size,...
    'StrongestFeatures', strongest_features, 'PointSelection',point_selection,...
    'Verbose',verbose);

%% encode the bag into features
feature_vectors = double(encode(bagVW, train_imds));

if verbose
    toc
end
%% Visualize Feature Vectors 
if visualize % visualization still doesn't work!
    img = read(train_imds(1), randi(train_imds(1).Count));
    featureVector = encode(bag, img);
    
    subplot(4,2,1); imshow(img);
    subplot(4,2,2); bar(featureVector);
    title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');
    
    img = read(train_imds(2), randi(train_imds(2).Count));
    featureVector = encode(bag, img);
    subplot(4,2,3); imshow(img);
    subplot(4,2,4); bar(featureVector);
    title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');
    
    img = read(train_imds(3), randi(train_imds(3).Count));
    featureVector = encode(bag, img);
    subplot(4,2,5); imshow(img);
    subplot(4,2,6); bar(featureVector);
    title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');
    
    img = read(train_imds(4), randi(train_imds(4).Count));
    featureVector = encode(bag, img);
    subplot(4,2,7); imshow(img);
    subplot(4,2,8); bar(featureVector);
    title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');
end
