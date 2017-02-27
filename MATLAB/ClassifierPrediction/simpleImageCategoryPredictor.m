% prediction of the category of random test tiles

%% setup parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

tile_sizes = [100];
tile_sizes_m = [80];
vocabulary_size = [50];

n = 1;
%num_datasets = length(tile_sizes);
tile_size = tile_sizes(n);
tile_size_m = tile_sizes_m(n);
str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];

sav_path_classifier = fullfile(root_dir, 'Results','Classification3Classes','Classifiers');
fname = fullfile(sav_path_classifier, ['trained_SURF_SVM_Classifier' num2str(vocabulary_size) '_' str '.mat']) ;

random_tiles_path = fullfile(root_dir, 'Results','Classification3Classes','TestTiles', str);
classes = {'BuiltUp'; 'NonBuiltUp'; 'Slum'};
classes_visnames = {'BU'; 'NBU';'S'};
num_classes = length(classes);
num_random_tiles_per_class = 10;

% the true labels
true_labels = [];
predicted_labels = [];

visualize = true;

%% Apply the pre-trained classfier
load(fname);

% get the filenames
for c = 1:num_classes
    class = char(classes{c});
    filenames{c} = dir(fullfile(random_tiles_path, class,'*.tif'));
end

% predict the categories
i =0;
if visualize
    figure('units','normalized','outerposition',[0 0 1 1]);
end
for c = 1:num_classes
    class = char(classes{c});
    fnames = {filenames{c}.name};
    paths = {filenames{c}.folder};
    for n = 1:num_random_tiles_per_class
        i = i + 1;
        true_labels{i} = class;
        % load test tile
        img_fname = fullfile(paths{n}, fnames{n});       
        img  = imread(img_fname);
        % predict
        [labelIdx, scores] = predict(categoryClassifier, img);
        predicted_labels{i} = char(categoryClassifier.Labels(labelIdx));
        if visualize
            subplot(3,num_random_tiles_per_class,i);
            imshow(img); axis on;
            title(['T: ' true_labels{i}]);
            xlabel(['Pr: ' predicted_labels{i}]);
        end
    end
end

true_labels = categorical(true_labels);
predicted_labels = categorical(predicted_labels);

%% evaluate
disp('Evaluating perfomance on the Random tiles');
perf_stats_rand = confusionmatStats(true_labels, predicted_labels);
perf_stats_rand.confusionMat
P = table( perf_stats_rand.accuracy*100, perf_stats_rand.sensitivity*100,...
    perf_stats_rand.specificity*100, perf_stats_rand.precision*100, ...
    perf_stats_rand.recall*100, perf_stats_rand.Fscore,...
    'RowNames', cellstr(classes),...
    'VariableNames', {'accuracy';'sensitivity'; 'specificity';...
    'precision';'recall';'Fscore'});
disp(P);