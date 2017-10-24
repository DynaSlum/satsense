function plot_fscore_all( fscores, tile_sizes, vocabulary_sizes, classes,...
    builtup_bar_color, nonbuiltup_bar_color, slum_bar_color, title_str, varargnin)

% optional arguments: xlabel_str, ylabel_str, zlabel_str, ...
% title_font, title_font_weight, title_font_size,...
% label_font, label_font_weight, label_font_size

% plot_fscore_all- function to generate bar plots of the fscore
%                       as a function of all tile and BoVW
%                       vocabulary sizes for all 3 semantic class
% for example usage see test_plot_performance_Bangalore


%% input parameter control
if (nargin < 1)
    error('The function plot_fscore_all expects at least 1 input parameters!');
end

%% default parameter values
npar = 2;
if nargin < npar|| isempty(tile_sizes)
    tile_sizes = [10 20 30 40 50 60];
end
npar = npar +1;
if nargin < npar || isempty(vocabulary_sizes)
    vocabulary_sizes = [10 20 50];
end
npar = npar +1;
if nargin < npar || isempty(classes)
    classes = {'BuiltUp';'NonBuiltUp';'Slum'};
end
npar = npar +1;
if nargin < npar || isempty(builtup_bar_color)
    builtup_bar_color = [0 0 1]; % blue
end
npar = npar +1;
if nargin < npar || isempty(nonbuiltup_bar_color)
    nonbuiltup_bar_color = [0 1 0]; % green
end
npar = npar +1;
if nargin < npar || isempty(slum_bar_color)
    slum_bar_color = [1 0 0]; % red
end
npar = npar +1;
if nargin < npar || isempty(title_str)
    title_str = 'F1 score';
end
% variable parameters
argn = npar;
if nargin > argn
    xlabel_str = varargin{argn+1};
else
    xlabel_str = 'Vocabulary size, V';
end
argn = argn + 1;
if nargin > argn
    ylabel_str = varargin{argn+1};
else
    ylabel_str = 'Tile size, N';
end
argn = argn + 1;
if nargin > argn
    zlabel_str = varargin{argn+1};
else
    zlabel_str = 'Value, [%]';
end
argn = argn + 1;
if nargin > argn
    title_font = varargin{argn+1};
else
    title_font= 'Helvetica';
end
argn = argn + 1;
if nargin > argn
    title_font_weight = varargin{argn+1};
else
    title_font_weight= 'bold';
end
argn = argn + 1;
if nargin > argn
    title_font_weight = varargin{argn+1};
else
    title_font_size= 24;
end
argn = argn + 1;
if nargin > argn
    label_font = varargin{argn+1};
else
    label_font= 'Helvetica';
end
argn = argn + 1;
if nargin > argn
    label_font_weight = varargin{argn+1};
else
    label_font_weight= 'bold';
end
argn = argn + 1;
if nargin > argn
    label_font_size = varargin{argn+1};
else
    label_font_size= 20;
end

%% axis parameters
x_ticks = 1:length(vocabulary_sizes);
x_tick_labels = vocabulary_sizes;
y_ticks = 1:length(tile_sizes);
y_tick_labels = tile_sizes;
z_ticks = 0:0.1:1;
z_tick_labels = ({'0','','0.2', '', '0.4','','0.6','','0.8','', '1'});

az = -40; el = 60;


%% plotting
data = fscores;
figure('units','normalized','outerposition',[0 0 1 1]);
for vs = 1:length(vocabulary_sizes)
    hold on;
    h = bar3(data(:,:,vs)', 'grouped');
    Xdat = get(h,'Xdata');
    for i=1:length(Xdat)
        Xdat{i}=Xdat{i}+(vs-1)*ones(size(Xdat{i}));
        set(h(i),'XData',Xdat{i});
    end
    set(h(1),'FaceColor', builtup_bar_color, 'EdgeColor', 'k');
    set(h(2),'FaceColor', nonbuiltup_bar_color, 'EdgeColor', 'k');
    set(h(3),'FaceColor', slum_bar_color, 'EdgeColor', 'k');
end

ax = gca;
set(ax,'XTick', x_ticks);
set(ax, 'XTickLabel',x_tick_labels, 'FontSize', label_font_size);
set(ax,'YTick', y_ticks);
set(ax, 'YTickLabel',y_tick_labels,'FontSize', label_font_size);
set(ax,'ZTick', z_ticks);
set(ax, 'ZTickLabel',z_tick_labels,'FontSize', label_font_size);
view(az, el);

axis on, grid on;
legend(classes, 'Location','northeast');

title(title_str, 'FontName', title_font, 'FontSize', title_font_size,...
    'FontWeight', title_font_weight);
xlabel(xlabel_str, 'FontName', label_font, 'FontSize', label_font_size,...
    'FontWeight', label_font_weight);
ylabel(ylabel_str, 'FontName', label_font, 'FontSize', label_font_size,...
    'FontWeight', label_font_weight);
zlabel(zlabel_str, 'FontName', label_font, 'FontSize', label_font_size,...
    'FontWeight', label_font_weight);


end


