function [ fig_h ] = plot_precission_recall_models_slum( data_table, ...
    prec_bar_color, recall_bar_color,...
    title_str, varargnin)

% optional arguments: xlabel_str, ylabel_str, ...
% title_font, title_font_weight, title_font_size,...       
% xlabel_rot_deg, label_font, label_font_weight, label_font_size

%plot_precission_recall_models_slum- function to generate bar plots of the 
%                       precission and recall of many classification models
%                       for the Slum class
%   The function expect a data_table as generated from importing a single
%   sheet of the Excel ClassificationPerformance book

%% default parameter values
if isempty(prec_bar_color)
    prec_bar_color = [1 0 0]; % red
end
if isempty(recall_bar_color)
    recall_bar_color = [0 0 1]; % blue
end
if isempty(title_str)
    title_str = 'Validation performance for class "Slum" for 200m tiles.'; 
end
if nargin > 4
    xlabel_str = varargin{1};
else
    xlabel_str = 'Classifier'; 
end
if nargin > 5
    ylabel_str = varargin{2};
else
    ylabel_str = 'Precission and Recall, [%]'; 
end
if nargin > 6
    title_font = varargin{3};
else
    title_font= 'Helvetica';
end
if nargin > 7
    title_font_weight = varargin{4};
else
    title_font_weight= 'bold';
end
if nargin > 8
    title_font_weight = varargin{5};
else
    title_font_size= 18;
end
if nargin > 9
    xlabel_rot_deg = varargin{6};
else
    xlabel_rot_deg = 30;
end
if nargin > 10
    label_font = varargin{7};
else
    label_font= 'Helvetica';
end
if nargin > 10
    label_font_weight = varargin{8};
else
    label_font_weight= 'normal';
end
if nargin > 11
    label_font_size = varargin{9};
else
    label_font_size= 12;
end

%% data_table --> variables
num_models = length(data_table.Model);
x_tick_labels = data_table.Namedescription;
y_ticks = [0:0.05:1];
y_tick_labels = [0:0.05:1]*100;

%% plotting
fig_h = figure('units','normalized','outerposition',[0 0 1 1]);

models = data_table.Model;
precission = data_table.PPVSlum;
recall = data_table.TPRSlum;
perf = [precission recall];
h = bar(models, perf);
set(h(1),'FaceColor', prec_bar_color, 'EdgeColor', 'k');
set(h(2),'FaceColor', recall_bar_color, 'EdgeColor', 'k');

axis on, grid on; legend('Precission', 'Recall', 'Location','northwest');
axis([0 num_models+1 0 1.05]);
title(title_str, 'FontName', title_font, 'FontSize', title_font_size,...
    'FontWeight', title_font_weight);
xlabel(xlabel_str, 'FontName', label_font, 'FontSize', label_font_size,...
    'FontWeight', label_font_weight);
ylabel(ylabel_str, 'FontName', label_font, 'FontSize', label_font_size,...
    'FontWeight', label_font_weight); 

%% making it pretty
ax = gca;
set(ax,'XTick', 1:num_models);
set(ax, 'XTickLabel',x_tick_labels);
set(ax, 'XTickLabelRotation',xlabel_rot_deg);
set(ax,'YTick', y_ticks);
set(ax, 'YTickLabel',y_tick_labels);
end

