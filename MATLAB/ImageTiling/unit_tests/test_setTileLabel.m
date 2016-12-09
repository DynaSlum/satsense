% Testing of setTileLabel

%% parameters
%tile_size = [333 333];
factor = 0.5;
data_path = 'C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\';

%% import masks
importfile(fullfile(data_path,'slums_municipality_raster_mask_8.tif'));
slum_mask = slums_municipality_raster_mask_8;
clear slums_municipality_raster_mask_8

importfile(fullfile(data_path,'urban_mask.tif'));

importfile(fullfile(data_path,'rural_mask.tif'));

%% run test cases
for i = 1:6
    switch i
        case 1
            % 1st test case: unifrom class Urban
            extent = [2332 2664 1000 1332];
            true_label =  'Urban';
        case 2
            % 2nd test case: mixture of Slum and Rural, mostly Slum
            extent = [3997 4329 1333 1665];
            true_label =  'Slum'; 
        case 3
            % mixture of Urban and Rural, mostly Rural
            extent = [3331 3663 667 999];
            true_label =  'Rural'; 
        case 4
            % mixture of Slum, Rural and Urban, mostly Slum
            extent = [5662 5994 3331 3663];
            true_label =  'Slum';            
        case 5
            % mixture of Slum, Rural and Urban
            extent = [5329 5661 3331 3663];
            true_label =  'Mixed';
        case 6
            % mixture of Slum and Urban, mostly Slum, non-square tile!
            extent = [2332 2664 4663 4872];
            true_label = 'Slum';
    end
    
    sr = extent(1); er = extent(2); sc = extent(3); ec = extent(4);
    tile_size(1) = er - sr +1;
    tile_size(2) = ec - sc +1;
    slum_submask  = slum_mask(sr:er, sc:ec,:);
    urban_submask = urban_mask(sr:er, sc:ec,:);
    rural_submask = rural_mask(sr:er, sc:ec,:);
    
    assigned_label = setTileLabel(tile_size, factor, ...
        slum_submask, urban_submask, rural_submask);
    
    
    if strcmp(assigned_label, true_label)
        disp(['The test with true class label ' true_label ' passed! ']);
    else
        disp(['The test with true class label ' true_label ' didn''t pass! ']);
    end
    
end
