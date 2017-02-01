function [predictedLabels, actualLabels, perf_stats] = ...
    class_perf_test(test_features_table, trained_classifier)

% class_perf_test - performance of a pre-trained classifier on the test dataset
% The inputs are the tesst dataset feature table and the pretrained
% classifier as output by the ClassifierLearner
% the output is the actual and predicted labels and classifictation perfomance 
% statistics as given by confusionmatStats

% classify the test data
predictedLabels = trained_classifier.predictFcn(test_features_table);

% get the actual labels
actualLabels = categorical(test_features_table.class_label);

% performance statistics
perf_stats = confusionmatStats(actualLabels, predictedLabels);
