% Testing of saveTile2File
data_path = 'C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\';
importfile(fullfile(data_path,'Mumbai_P4_R1C1_3_clipped.tif'));
extent = [5662 5994 3664 3996];
sr = extent(1); er = extent(2); sc = extent(3); ec = extent(4);
image_data = Mumbai_P4_R1C1_3_clipped(sr:er, sc:ec,:);
label =  'Slum';
dataset_path = 'C:\Projects\DynaSlum\Data\Kalyan\Datasets\px333m200';
base_fname = 'Mumbai_P4_R1C1_3_clipped'; ext = 'tif';
saveTile2File(image_data,  extent, dataset_path, label, base_fname, ext);