function [ fig_h ] = plot_precission_recall_best_model_3classes( data_table, ...
    best_model_idx, prec_bar_color, recall_bar_color,...
    title_str, varargnin)

% optional arguments: xlabel_str, ylabel_str, ...
% title_font, title_font_weight, title_font_size,...       
% label_font, label_font_weight, label_font_size

%plot_precission_recall_best_model_3classes- function to generate bar plots of the 
%                       precission and recall of the best training classification models
%                       for the 3 classes: BuildUp, NonBuildUp and Slum
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
    title_str = 'Validation performance best classifier for 200m tiles.'; 
end
if nargin > 5
    xlabel_str = varargin{1};
else
    xlabel_str = ['3-class model: '  data_table.Namedescription(best_model_idx)]; 
end
if nargin > 6
    ylabel_str = varargin{2};
else
    ylabel_str = ['Precission and Recall, [%] ']; 
end
if nargin > 7
    title_font = varargin{3};
else
    title_font= 'Helvetica';
end
if nargin > 8
    title_font_weight = varargin{4};
else
    title_font_weight= 'bold';
end
if nargin > 9
    title_font_weight = varargin{5};
else
    title_font_size= 16;
end
if nargin > 10
    label_font = varargin{6};
else
    label_font= 'Helvetica';
end
if nargin > 11
    label_font_weight = varargin{7};
else
    label_font_weight= 'normal';
end
if nargin > 12
    label_font_size = varargin{8};
else
    label_font_size= 12;
end

%% data_table --> variables
num_classes = 3;
x_tick_labels = {'BuildUp', 'NonBuildUp', 'Slum'};
y_ticks = [0:0.05:1];
y_tick_labels = [0:0.05:1]*100;

%% plotting
fig_h = figure('units','normalized','outerposition',[0 0 1 1]);

precissions = [data_table.PPVBU(best_model_idx); data_table.PPVNBU(best_model_idx); data_table.PPVSlum(best_model_idx)];
recalls = [data_table.TPRBU(best_model_idx); data_table.TPRNBU(best_model_idx); data_table.TPRSlum(best_model_idx)];
perf = [precissions recalls];
h = bar(perf);
set(h(1),'FaceColor', prec_bar_color, 'EdgeColor', 'k');
set(h(2),'FaceColor', recall_bar_color, 'EdgeColor', 'k');

axis on, grid on; legend('Precission', 'Recall', 'Location','northwest');
axis([0 num_classes+1 0 1.05]);
title(title_str, 'FontName', title_font, 'FontSize', title_font_size,...
    'FontWeight', title_font_weight);
xlabel(xlabel_str, 'FontName', label_font, 'FontSize', label_font_size,...
    'FontWeight', label_font_weight);
ylabel(ylabel_str, 'FontName', label_font, 'FontSize', label_font_size,...
    'FontWeight', label_font_weight); 

%% making it pretty
ax = gca;
set(ax,'XTick', 1:num_classes);
set(ax, 'XTickLabel',x_tick_labels);
set(ax,'YTick', y_ticks);
set(ax, 'YTickLabel',y_tick_labels);
end

