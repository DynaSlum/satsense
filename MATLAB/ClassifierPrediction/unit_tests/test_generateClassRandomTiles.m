% Testing of slumTiling and nonSlumTiling

%% parameters
if isunix
    root_dir = fullfile('/home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end

sav_path = fullfile(root_dir, 'Results', 'Classification3Classes', 'TestTiles');
factor = 0.8;
tile_sizes = 100;
tile_sizes_m = 80;

num_random_tiles_per_class = 10;
classes = {'Slum'; 'BuiltUp'; 'NonBuiltUp'};
num_classes = length(classes);
num_datasets = length(tile_sizes);

data_path = fullfile(root_dir, 'Data','Kalyan','Rasterized_Lourens');
image_fullfname = fullfile(data_path, 'Mumbai_P4_R1C1_3_clipped_rgb.tif');
slum_mask = fullfile(data_path,'all_slums.tif');
builtup_mask = fullfile(data_path,'builtup_mask.tif');
nonbuiltup_mask = fullfile(data_path,'nonbuiltup_mask.tif');

masks = {slum_mask; builtup_mask; nonbuiltup_mask};

%% for all datasets get random tiles
for d = 1: num_datasets
    tile_size = tile_sizes(d);
    tile_size_m = tile_sizes_m(d);
    
    str = ['px' num2str(tile_size) 'm' num2str(tile_size_m)];
    tiles_path = fullfile(sav_path, str);
    
    disp(['Getting random tiles for dataset ' str]);
    
    % get random tiles
    for c = 1: num_classes
        label = char(classes{c});
        disp(['Getting random tiles for class ' label]);
        
        mask_fullfname = char(masks{c});
        
        tiles_names{d,c} = generateClassRandomTiles( image_fullfname, label, ...
           num_random_tiles_per_class, [tile_size tile_size], factor, mask_fullfname, tiles_path);
              
        
        disp('Done!');
        disp('---------------------------------');
    end
end

%% visualize
figure('units','normalized','outerposition',[0 0 1 1]);
sbp = 0;
for d = 1: num_datasets 
    for c = 1: num_classes
        label = char(classes{c});
        for i = 1: num_random_tiles_per_class
            sbp = sbp + 1;
            tile = imread(tiles_names{d,c}{i});
            subplot(3,num_random_tiles_per_class,sbp);
            imshow(tile); axis on; title(['Test tile: ' label]);
        end
    end
end
 


disp('DONE!');
