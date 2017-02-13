% Testing of saveTile2File
if isunix
    root_dir = fullfile('\home','elena','DynaSlum');
else
    root_dir = fullfile('C:','Projects', 'DynaSlum');
end
data_path = fullfile(root_dir, 'Data','Kalyan','Rasterized_Lourens');
importfile(fullfile(data_path,'Mumbai_P4_R1C1_3_clipped_rgb.tif'));
extent = [5662 5994 3664 3996];
sr = extent(1); er = extent(2); sc = extent(3); ec = extent(4);
image_data = Mumbai_P4_R1C1_3_clipped_rgb(sr:er, sc:ec,:);
label =  'Slum';

dataset_path = fullfile(root_dir, 'Data','Kalyan','Datasets4ClassesInclMixed','px333m200');
base_fname = 'Mumbai_P4_R1C1_3_clipped_rgb'; ext = 'tif';
[~] = saveTile2File(image_data,  extent, dataset_path, label, base_fname, ext);