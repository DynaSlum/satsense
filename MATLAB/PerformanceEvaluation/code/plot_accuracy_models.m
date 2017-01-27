function [ fig_h ] = plot_accuracy_models( data_table, bar_color, ...
    title_str, varargin)

% optional arguments: xlabel_str, ylabel_str, ...
% title_font, title_font_weight, title_font_size,...       
% xlabel_rot_deg, label_font, label_font_weight, label_font_size

%plot_accuracy_models- function to generate a bar plot of the accuracy of 
%                       many classification models
%   The function expect a data_table as generated from importing a single
%   sheet of the Excel ClassificationPerformance book

%% default parameter values
if isempty(bar_color)
    bar_color = [0 1 0]; % green
end
if isempty(title_str)
    title_str = 'Validation performance for tiles of size 200m.'; 
end
if nargin > 3
    xlabel_str = varargin{1};
else
    xlabel_str = 'Classifier'; 
end
if nargin > 4
    ylabel_str = varargin{2};
else
    ylabel_str = 'Accuracy, [%]'; 
end
if nargin > 5
    title_font = varargin{3};
else
    title_font= 'Helvetica';
end
if nargin > 6
    title_font_weight = varargin{4};
else
    title_font_weight= 'bold';
end
if nargin > 7
    title_font_weight = varargin{5};
else
    title_font_size= 18;
end
if nargin > 8
    xlabel_rot_deg = varargin{6};
else
    xlabel_rot_deg = 30;
end
if nargin > 9
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
    label_font_size = varargin{8};
else
    label_font_size= 12;
end

%% data_table --> variables
num_models = length(data_table.Model);
x_tick_labels = data_table.Namedescription;
y_tick_labels = [0.55:0.05:1]*100;

%% plotting
fig_h = figure;

models = data_table.Model;
accuracy = data_table.ACC;
bar(models, accuracy, 'FaceColor', bar_color, 'EdgeColor', bar_color);

axis on, grid on; 
axis([0 num_models+1 0.55 1]);
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
set(ax, 'YTickLabel',y_tick_labels);
end

